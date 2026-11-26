// SPDX-License-Identifier: Apache-2.0

use anyhow::Context;
use hypervisor::arch::x86::gdt::{gdt_entry, segment_from_gdt};
use vm_memory::{Bytes, GuestAddress};

use crate::GuestMemoryMmap;
use crate::layout::{MP_WAKEUP_MAILBOX, MP_WAKEUP_START};

const PAGE_SIZE: u64 = 4096;
const PML4: u64 = MP_WAKEUP_START.0 + 2 * PAGE_SIZE;
const PDPT: u64 = PML4 + PAGE_SIZE;
const PD: u64 = PDPT + PAGE_SIZE;
const GDT: u64 = MP_WAKEUP_START.0 + 0x100;
const SIZE: usize = 5 * PAGE_SIZE as usize;

// This code runs only inside the guest, using private memory and pre-seal state.
// RDI is the VP's APIC ID and RSI points to the ACPI wakeup mailbox.
core::arch::global_asm!(
    r#"
    .pushsection .rodata.chv_mp_wakeup,"a",@progbits
    .global chv_mp_wakeup_start
    .global chv_mp_wakeup_end
chv_mp_wakeup_start:
    cli
2:
    cmp word ptr [rsi], 1
    jne 3f
    cmp dword ptr [rsi + 4], edi
    jne 3f
    mov rax, qword ptr [rsi + 8]
    mov word ptr [rsi], 0
    jmp rax
3:
    pause
    jmp 2b
chv_mp_wakeup_end:
    .popsection
"#
);

unsafe extern "C" {
    static chv_mp_wakeup_start: u8;
    static chv_mp_wakeup_end: u8;
}

pub fn madt_entry() -> [u8; 16] {
    let mut entry = [0; 16];
    entry[0] = 16; // ACPI MADT Multiprocessor Wakeup, version 0.
    entry[1] = entry.len() as u8;
    entry[8..].copy_from_slice(&MP_WAKEUP_MAILBOX.0.to_le_bytes());
    entry
}

pub fn prepare(memory: &GuestMemoryMmap) -> anyhow::Result<()> {
    memory
        .write_slice(&[0; SIZE], MP_WAKEUP_START)
        .context("initialize private AP firmware and mailbox")?;
    let start = std::ptr::addr_of!(chv_mp_wakeup_start);
    let len = std::ptr::addr_of!(chv_mp_wakeup_end) as usize - start as usize;
    anyhow::ensure!(len <= 0x100, "AP waiting code overlaps its GDT");
    // SAFETY: both labels delimit one immutable, relocation-free assembly blob.
    let code = unsafe { std::slice::from_raw_parts(start, len) };
    memory.write_slice(code, MP_WAKEUP_START)?;
    let gdt = [
        0,
        gdt_entry(0xa09b, 0, 0xffffffff),
        gdt_entry(0xc093, 0, 0xffffffff),
        gdt_entry(0x008b, 0, 0x67),
    ];
    for (index, entry) in gdt.into_iter().enumerate() {
        memory.write_obj(entry, GuestAddress(GDT + index as u64 * 8))?;
    }
    memory.write_obj(PDPT | 3, GuestAddress(PML4))?;
    memory.write_obj(PD | 3, GuestAddress(PDPT))?;
    for index in 0..512_u64 {
        memory.write_obj((index << 21) | 0x83, GuestAddress(PD + index * 8))?;
    }
    Ok(())
}

pub fn configure_vcpu(vcpu: &dyn hypervisor::Vcpu, apic_id: u32) -> anyhow::Result<()> {
    let mut sregs = vcpu.get_sregs()?;
    sregs.cr0 = 0x8000_0011; // PE, ET, PG.
    sregs.cr3 = PML4;
    sregs.cr4 = 1 << 5; // PAE.
    // Linux's 64-bit trampoline installs page tables containing NX entries.
    sregs.efer = (1 << 8) | (1 << 10) | (1 << 11); // LME, LMA, NXE.
    sregs.cs = segment_from_gdt(gdt_entry(0xa09b, 0, 0xffffffff), 1);
    sregs.gdt.base = GDT;
    sregs.gdt.limit = 31;
    vcpu.set_sregs(&sregs)?;
    let mut regs = vcpu.create_standard_regs();
    regs.set_rflags(2);
    regs.set_rip(MP_WAKEUP_START.0);
    regs.set_rdi(u64::from(apic_id));
    regs.set_rsi(MP_WAKEUP_MAILBOX.0);
    vcpu.set_regs(&regs)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vm_memory::Address;

    #[test]
    fn private_firmware_preserves_neighbors_and_identity_maps_entry_memory() {
        let memory = GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x100000)]).unwrap();
        let before = MP_WAKEUP_START.checked_sub(1).unwrap();
        let after = MP_WAKEUP_START.unchecked_add(SIZE as u64);
        memory.write_obj(0xa5_u8, before).unwrap();
        memory.write_obj(0x5a_u8, after).unwrap();
        memory.write_obj(1_u16, MP_WAKEUP_MAILBOX).unwrap();
        prepare(&memory).unwrap();
        assert_eq!(memory.read_obj::<u8>(before).unwrap(), 0xa5);
        assert_eq!(memory.read_obj::<u8>(after).unwrap(), 0x5a);
        assert_eq!(memory.read_obj::<u16>(MP_WAKEUP_MAILBOX).unwrap(), 0);

        for address in [MP_WAKEUP_START.0, MP_WAKEUP_MAILBOX.0, GDT, 0x10000] {
            let pml4e: u64 = memory
                .read_obj(GuestAddress(PML4 + ((address >> 39) & 511) * 8))
                .unwrap();
            let pdpte: u64 = memory
                .read_obj(GuestAddress((pml4e & !4095) + ((address >> 30) & 511) * 8))
                .unwrap();
            let pde: u64 = memory
                .read_obj(GuestAddress((pdpte & !4095) + ((address >> 21) & 511) * 8))
                .unwrap();
            assert_eq!(pml4e & 3, 3);
            assert_eq!(pdpte & 3, 3);
            assert_eq!(pde & 0x83, 0x83);
            assert_eq!((pde & !0x1fffff) | (address & 0x1fffff), address);
        }
    }
}
