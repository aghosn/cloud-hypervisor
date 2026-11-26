// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause
//
// MSHV-style MMIO instruction emulation for the Themis hypervisor backend.
//
// Instead of decoding MMIO instructions in the capavisor (which only handled
// a handful of opcodes), we use CHV's existing iced-x86 emulator infrastructure
// (MOV, MOVZX, CMP, MOVS, STOS, OR, etc.) to emulate the faulting instruction
// in userspace. Instructions are fetched through CHV's own guest-memory access;
// the capavisor supplies only exit metadata, never guest instruction bytes.
//
// See P16.6i in todo.md for the full design rationale.

use anyhow::anyhow;
use iced_x86::{Decoder, DecoderError, DecoderOptions};
use log::debug;

use super::ThemisVcpu;
use crate::arch::emulator::{PlatformEmulator, PlatformError};
use crate::arch::x86::emulator::EmulatorCpuState;
use crate::cpu::Vcpu;

/// Emulator context for a single MMIO EPT-violation exit.
///
/// Created per-exit with the faulting GPA. Implements `PlatformEmulator` so it can be
/// passed to `Emulator::emulate_first_insn()`.
pub struct ThemisEmulatorContext<'a> {
    pub vcpu: &'a ThemisVcpu,
    pub mmio_gpa: u64,
    /// Mask with the VTOM bit set (e.g. `1 << 38`).  CoCo guests set this bit
    /// in page-table entries for shared (MMIO / DMA) pages.  We strip it from
    /// GPAs before passing them to the device model so that addresses match the
    /// canonical device-model layout.
    pub vtom_mask: u64,
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

        let sregs = self
            .vcpu
            .get_sregs()
            .map_err(|e| PlatformError::TranslateVirtualAddress(e.into()))?;
        let cr3 = sregs.cr3;
        if sregs.cr0 & (1 << 31) == 0 {
            return Ok(gva & !self.vtom_mask);
        }
        if sregs.efer & (1 << 10) == 0 || sregs.cr4 & (1 << 12) != 0 {
            return Err(PlatformError::TranslateVirtualAddress(anyhow!(
                "Themis MMIO emulation requires four-level long-mode paging"
            )));
        }

        let read_entry = |table_phys: u64, index: u64| -> Result<u64, PlatformError> {
            let addr = (table_phys & PHYS_ADDR_MASK) | (index * 8);
            let mut buf = [0u8; 8];
            let count = vm_ops
                .guest_mem_read(addr, &mut buf)
                .map_err(|e| PlatformError::TranslateVirtualAddress(e.into()))?;
            if count != buf.len() {
                return Err(PlatformError::TranslateVirtualAddress(anyhow!(
                    "short page-table read"
                )));
            }
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
            return Ok(((pdpte & 0x000F_FFFF_C000_0000) | (gva & 0x3FFF_FFFF)) & !self.vtom_mask);
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
            return Ok(((pde & 0x000F_FFFF_FFE0_0000) | (gva & 0x1F_FFFF)) & !self.vtom_mask);
        }

        // PT
        let pte = read_entry(pde, (gva >> 12) & 0x1FF)?;
        if pte & PAGE_PRESENT == 0 {
            return Err(PlatformError::TranslateVirtualAddress(anyhow!(
                "PTE not present for GVA {gva:#x}"
            )));
        }

        Ok(((pte & PHYS_ADDR_MASK) | (gva & 0xFFF)) & !self.vtom_mask)
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
        if self
            .vcpu
            .vm_state
            .ioevents
            .lock()
            .unwrap()
            .write(crate::IoEventAddress::Mmio(gpa), data)
            .map_err(|e| PlatformError::MemoryWriteFailure(e.into()))?
        {
            return Ok(());
        }
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

    pub(super) fn instruction_bytes(&self, ip: u64) -> Result<[u8; 16], PlatformError> {
        let vm_ops = self
            .vcpu
            .vm_ops()
            .ok_or_else(|| PlatformError::MemoryReadFailure(anyhow!("no vm_ops")))?;
        let sregs = self
            .vcpu
            .get_sregs()
            .map_err(|e| PlatformError::GetCpuStateFailure(e.into()))?;
        if sregs.cs.l == 0 || sregs.efer & (1 << 10) == 0 {
            return Err(PlatformError::MemoryReadFailure(anyhow!(
                "Themis instruction emulator requires 64-bit code"
            )));
        }
        fetch_instruction(ip, |gva, bytes| {
            let gpa = self.translate_gva(gva)?;
            let count = vm_ops
                .guest_mem_read(gpa, bytes)
                .map_err(|e| PlatformError::MemoryReadFailure(e.into()))?;
            if count != bytes.len() {
                return Err(PlatformError::MemoryReadFailure(anyhow!(
                    "short instruction read"
                )));
            }
            Ok(())
        })
    }
}

fn fetch_instruction(
    ip: u64,
    mut read: impl FnMut(u64, &mut [u8]) -> Result<(), PlatformError>,
) -> Result<[u8; 16], PlatformError> {
    let mut bytes = [0; 16];
    let first = (0x1000 - (ip & 0xfff) as usize).min(15);
    read(ip, &mut bytes[..first])?;
    let mut decoder = Decoder::with_ip(64, &bytes[..first], ip, DecoderOptions::NONE);
    let _ = decoder.decode();
    if decoder.last_error() == DecoderError::NoMoreBytes && first < 15 {
        let next = ip.checked_add(first as u64).ok_or_else(|| {
            PlatformError::MemoryReadFailure(anyhow!("instruction address overflow"))
        })?;
        read(next, &mut bytes[first..15])?;
        decoder = Decoder::with_ip(64, &bytes[..15], ip, DecoderOptions::NONE);
        let _ = decoder.decode();
    }
    if decoder.last_error() != DecoderError::None {
        return Err(PlatformError::MemoryReadFailure(anyhow!(
            "invalid or incomplete instruction"
        )));
    }
    Ok(bytes)
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

    fn fetch(&self, ip: u64, instruction_bytes: &mut [u8]) -> Result<(), PlatformError> {
        let bytes = self.instruction_bytes(ip)?;
        let source = bytes.get(..instruction_bytes.len()).ok_or_else(|| {
            PlatformError::MemoryReadFailure(anyhow!("instruction buffer too large"))
        })?;
        instruction_bytes.copy_from_slice(source);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_instruction_does_not_read_an_unmapped_next_page() {
        let mut reads = Vec::new();
        let bytes = fetch_instruction(0xfff, |addr, bytes| {
            reads.push(addr);
            assert_eq!(addr, 0xfff);
            bytes[0] = 0x90;
            Ok(())
        })
        .unwrap();
        assert_eq!(bytes[0], 0x90);
        assert_eq!(reads, vec![0xfff]);
    }

    #[test]
    fn cross_page_instruction_is_fetched_from_both_pages() {
        let bytes = fetch_instruction(0xffe, |addr, bytes| {
            match addr {
                0xffe => bytes.copy_from_slice(&[0x48, 0x8b]),
                0x1000 => bytes[0] = 0x00,
                _ => panic!("unexpected fetch address"),
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(&bytes[..3], &[0x48, 0x8b, 0x00]);
    }

    #[test]
    fn inaccessible_instruction_memory_returns_an_error() {
        let result = fetch_instruction(0xfff, |addr, bytes| {
            if addr == 0xfff {
                bytes[0] = 0x48;
                Ok(())
            } else {
                Err(PlatformError::MemoryReadFailure(anyhow!("unmapped")))
            }
        });
        result.unwrap_err();
    }
}
