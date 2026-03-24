// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause
//
// MSHV-style MMIO instruction emulation for the Themis hypervisor backend.
//
// Instead of decoding MMIO instructions in the capavisor (which only handled
// a handful of opcodes), we use CHV's existing iced-x86 emulator infrastructure
// (MOV, MOVZX, CMP, MOVS, STOS, OR, etc.) to emulate the faulting instruction
// in userspace.  The capavisor only needs to supply the instruction bytes and
// the faulting GPA; all decode, register update, and RIP advancement happens here.
//
// See P16.6i in todo.md for the full design rationale.

use anyhow::anyhow;
use log::debug;

use crate::arch::emulator::{PlatformEmulator, PlatformError};
use crate::arch::x86::emulator::EmulatorCpuState;
use crate::cpu::Vcpu;

use super::ThemisVcpu;

/// Emulator context for a single MMIO EPT-violation exit.
///
/// Created per-exit with the faulting GPA and the instruction bytes from the
/// capavisor intercept message.  Implements `PlatformEmulator` so it can be
/// passed to `Emulator::emulate_first_insn()`.
pub struct ThemisEmulatorContext<'a> {
    pub vcpu: &'a ThemisVcpu,
    pub mmio_gpa: u64,
    pub insn_bytes: [u8; 16],
}

// ---------------------------------------------------------------------------
// Guest page-table walker (4-level x86-64 paging)
// ---------------------------------------------------------------------------

const PAGE_PRESENT: u64 = 1 << 0;
const PAGE_SIZE_BIT: u64 = 1 << 7; // PS bit — 2 MB or 1 GB page
const PHYS_ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000; // bits 51:12

impl ThemisEmulatorContext<'_> {
    /// Translate a guest virtual address to a guest physical address by walking
    /// the guest's CR3 page tables through `vm_ops.guest_mem_read`.
    fn translate_gva(&self, gva: u64) -> Result<u64, PlatformError> {
        let vm_ops = self
            .vcpu
            .vm_ops()
            .ok_or_else(|| PlatformError::TranslateVirtualAddress(anyhow!("no vm_ops")))?;

        let cr3 = self
            .vcpu
            .get_sregs()
            .map_err(|e| PlatformError::TranslateVirtualAddress(e.into()))?
            .cr3;

        let read_entry = |table_phys: u64, index: u64| -> Result<u64, PlatformError> {
            let addr = (table_phys & PHYS_ADDR_MASK) | (index * 8);
            let mut buf = [0u8; 8];
            vm_ops
                .guest_mem_read(addr, &mut buf)
                .map_err(|e| PlatformError::TranslateVirtualAddress(e.into()))?;
            Ok(u64::from_le_bytes(buf))
        };

        // PML4
        let pml4e = read_entry(cr3, (gva >> 39) & 0x1FF)?;
        if pml4e & PAGE_PRESENT == 0 {
            return Err(PlatformError::TranslateVirtualAddress(anyhow!(
                "PML4E not present for GVA {gva:#x}"
            )));
        }

        // PDPT
        let pdpte = read_entry(pml4e, (gva >> 30) & 0x1FF)?;
        if pdpte & PAGE_PRESENT == 0 {
            return Err(PlatformError::TranslateVirtualAddress(anyhow!(
                "PDPTE not present for GVA {gva:#x}"
            )));
        }
        if pdpte & PAGE_SIZE_BIT != 0 {
            // 1 GB page
            return Ok((pdpte & 0x000F_FFFF_C000_0000) | (gva & 0x3FFF_FFFF));
        }

        // PD
        let pde = read_entry(pdpte, (gva >> 21) & 0x1FF)?;
        if pde & PAGE_PRESENT == 0 {
            return Err(PlatformError::TranslateVirtualAddress(anyhow!(
                "PDE not present for GVA {gva:#x}"
            )));
        }
        if pde & PAGE_SIZE_BIT != 0 {
            // 2 MB page
            return Ok((pde & 0x000F_FFFF_FFE0_0000) | (gva & 0x1F_FFFF));
        }

        // PT
        let pte = read_entry(pde, (gva >> 12) & 0x1FF)?;
        if pte & PAGE_PRESENT == 0 {
            return Err(PlatformError::TranslateVirtualAddress(anyhow!(
                "PTE not present for GVA {gva:#x}"
            )));
        }

        Ok((pte & PHYS_ADDR_MASK) | (gva & 0xFFF))
    }

    /// Read from guest memory at a guest virtual address.
    /// Translates GVA→GPA, then tries RAM; if that fails, falls back to MMIO.
    fn read_gva(&self, gva: u64, data: &mut [u8]) -> Result<(), PlatformError> {
        let gpa = self.translate_gva(gva)?;
        let vm_ops = self
            .vcpu
            .vm_ops()
            .ok_or_else(|| PlatformError::MemoryReadFailure(anyhow!("no vm_ops")))?;

        debug!(
            "themis emulator: read {} bytes [{gva:#x} -> {gpa:#x}]",
            data.len()
        );

        if vm_ops.guest_mem_read(gpa, data).is_err() {
            vm_ops
                .mmio_read(gpa, data)
                .map_err(|e| PlatformError::MemoryReadFailure(e.into()))?;
        }
        Ok(())
    }

    /// Write to guest memory at a guest virtual address.
    fn write_gva(&self, gva: u64, data: &[u8]) -> Result<(), PlatformError> {
        let gpa = self.translate_gva(gva)?;
        let vm_ops = self
            .vcpu
            .vm_ops()
            .ok_or_else(|| PlatformError::MemoryWriteFailure(anyhow!("no vm_ops")))?;

        debug!(
            "themis emulator: write {} bytes [{gva:#x} -> {gpa:#x}]",
            data.len()
        );

        if vm_ops.guest_mem_write(gpa, data).is_err() {
            vm_ops
                .mmio_write(gpa, data)
                .map_err(|e| PlatformError::MemoryWriteFailure(e.into()))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PlatformEmulator implementation
// ---------------------------------------------------------------------------

impl PlatformEmulator for ThemisEmulatorContext<'_> {
    type CpuState = EmulatorCpuState;

    fn read_memory(&self, gva: u64, data: &mut [u8]) -> Result<(), PlatformError> {
        // Handle cross-page reads (at most one page boundary).
        let pg1 = gva >> 12;
        let pg2 = (gva + data.len() as u64 - 1) >> 12;
        if pg1 != pg2 {
            let split = ((pg2 << 12) - gva) as usize;
            self.read_gva(gva, &mut data[..split])?;
            self.read_gva(gva + split as u64, &mut data[split..])?;
            return Ok(());
        }
        self.read_gva(gva, data)
    }

    fn write_memory(&mut self, gva: u64, data: &[u8]) -> Result<(), PlatformError> {
        let pg1 = gva >> 12;
        let pg2 = (gva + data.len() as u64 - 1) >> 12;
        if pg1 != pg2 {
            let split = ((pg2 << 12) - gva) as usize;
            self.write_gva(gva, &data[..split])?;
            self.write_gva(gva + split as u64, &data[split..])?;
            return Ok(());
        }
        self.write_gva(gva, data)
    }

    fn cpu_state(&self, _cpu_id: usize) -> Result<Self::CpuState, PlatformError> {
        let regs = self
            .vcpu
            .get_regs()
            .map_err(|e| PlatformError::GetCpuStateFailure(e.into()))?;
        let sregs = self
            .vcpu
            .get_sregs()
            .map_err(|e| PlatformError::GetCpuStateFailure(e.into()))?;
        Ok(EmulatorCpuState { regs, sregs })
    }

    fn set_cpu_state(&self, _cpu_id: usize, state: Self::CpuState) -> Result<(), PlatformError> {
        self.vcpu
            .set_regs(&state.regs)
            .map_err(|e| PlatformError::SetCpuStateFailure(e.into()))?;
        self.vcpu
            .set_sregs(&state.sregs)
            .map_err(|e| PlatformError::SetCpuStateFailure(e.into()))?;
        Ok(())
    }

    fn fetch(&self, _ip: u64, instruction_bytes: &mut [u8]) -> Result<(), PlatformError> {
        // The capavisor already read 16 instruction bytes from guest memory
        // and placed them in the intercept message.  Return those directly.
        let n = instruction_bytes.len().min(self.insn_bytes.len());
        instruction_bytes[..n].copy_from_slice(&self.insn_bytes[..n]);
        Ok(())
    }
}
