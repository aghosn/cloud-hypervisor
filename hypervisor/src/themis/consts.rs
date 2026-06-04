//! Constants for the Themis backend.
//!
//! THHV ABI flags, VMX exit reasons, APIC/ICR/MSI bit fields, etc.
//!
//! Items here are scoped `pub(super)` and re-exported / consumed by sibling
//! submodules under `crate::themis`.
//!
//! Constants that are part of the shared Themis ABI are re-exported from
//! `themis-abi` (the single source of truth shared with capavisor, thhv, and
//! libthemis) rather than re-defined here.

use crate::arch::x86::MsrEntry;

// ── Re-exports from the shared `themis-abi` crate ────────────────────────── //
// VpRegister enum (used as u64 register-id over the THHV ioctl wire).
pub(super) use themis_abi::regs::{ALL_VP_REGISTERS, VpRegister};
// Synthetic exit reason injected by capavisor (high bit set, not an SDM value).
pub(super) use themis_abi::synthetic_exits::THEMIS_EXIT_DOORBELL;
// Domain-comm message types.
pub(super) use themis_abi::regs::THEMIC_MSG_SHUTDOWN;
// SIPI real-mode segment access rights (VMX encoding).
pub(super) use themis_abi::regs::access_rights::CODE16 as REALMODE_CODE_SEG_AR;
pub(super) use themis_abi::regs::access_rights::DATA16 as REALMODE_DATA_SEG_AR;

pub(super) const THHV_IOCTL_MAGIC: u8 = 0xB8;
pub(super) const THHV_SCHED_SYNC: u32 = 0;
pub(super) const THHV_MEM_F_UNMAP: u32 = 1 << 0;
pub(super) const THHV_MEM_F_ALIAS: u32 = 1 << 1;
pub(super) const THHV_MEM_R_READ: u32 = 1 << 0;
pub(super) const THHV_MEM_R_WRITE: u32 = 1 << 1;
pub(super) const THHV_MEM_R_EXEC: u32 = 1 << 2;
pub(super) const THHV_IRQFD_FLAG_DEASSIGN: u32 = 1 << 0;
pub(super) const THHV_IOEVENTFD_FLAG_DATAMATCH: u32 = 1 << 0;
pub(super) const THHV_IOEVENTFD_FLAG_PIO: u32 = 1 << 1;
pub(super) const THHV_IOEVENTFD_FLAG_DEASSIGN: u32 = 1 << 2;
pub(super) const THHV_META_PAGES_PER_VP: usize = 3;
pub(super) const THHV_META_PAGES_SHARED: usize = 4;
pub(super) const THHV_QUERY_META_PAGES_PER_VP: u32 = 1;
pub(super) const THHV_QUERY_META_PAGES_SHARED: u32 = 2;

pub(super) const EXIT_REASON_EXCEPTION_NMI: u32 = 0;
pub(super) const EXIT_REASON_EXTERNAL_INTERRUPT: u32 = 1;
pub(super) const EXIT_REASON_TRIPLE_FAULT: u32 = 2;
pub(super) const EXIT_REASON_CPUID: u32 = 10;
pub(super) const EXIT_REASON_HLT: u32 = 12;
pub(super) const EXIT_REASON_VMCALL: u32 = 18;
pub(super) const EXIT_REASON_CR_ACCESS: u32 = 28;     // Intel SDM: MOV to/from CR0/CR3/CR4/CR8, CLTS, LMSW
pub(super) const EXIT_REASON_IO_INSTRUCTION: u32 = 30; // Intel SDM: IN, INS, OUT, OUTS
pub(super) const EXIT_REASON_RDMSR: u32 = 31;
pub(super) const EXIT_REASON_WRMSR: u32 = 32;
pub(super) const EXIT_REASON_EPT_VIOLATION: u32 = 48;
pub(super) const EXIT_REASON_APIC_ACCESS: u32 = 44;  // Intel SDM: APIC-access MMIO trap

// EPT-violation exit-qualification bits (Intel SDM Vol 3C §28.2.1).
// Only EXECUTE is consumed today; add READ/WRITE here if a future handler needs them.
pub(super) const EPT_VIOLATION_EXECUTE: u64 = 1 << 2;

// ── APIC register offsets (Intel SDM Vol 3A §10.4.1) ──────────────────────
pub(super) const APIC_REG_ICR_LOW: u32 = 0x300;
pub(super) const APIC_REG_ICR_HIGH: u32 = 0x310;
pub(super) const APIC_REG_EOI: u32 = 0x0B0;

// ── ICR bit fields (Intel SDM Vol 3A §10.6.1) ────────────────────────────
pub(super) const ICR_VECTOR_MASK: u32 = 0xFF;
pub(super) const ICR_DELIVERY_MODE_SHIFT: u32 = 8;
pub(super) const ICR_DELIVERY_MODE_MASK: u32 = 0x7;
pub(super) const ICR_DEST_SHORTHAND_SHIFT: u32 = 18;
pub(super) const ICR_DEST_SHORTHAND_MASK: u32 = 0x3;
pub(super) const ICR_HIGH_DEST_SHIFT: u32 = 24;
pub(super) const ICR_HIGH_DEST_MASK: u32 = 0xFF;

// ICR delivery modes
pub(super) const ICR_MODE_FIXED: u32 = 0;
pub(super) const ICR_MODE_LOWEST_PRIORITY: u32 = 1;
pub(super) const ICR_MODE_INIT: u32 = 5;
pub(super) const ICR_MODE_SIPI: u32 = 6;

// MSI address field (Intel SDM Vol 3A §10.11.1)
pub(super) const MSI_ADDR_DEST_ID_SHIFT: u32 = 12;
pub(super) const MSI_ADDR_DEST_ID_MASK: u32 = 0xFF;

// Policy-kind discriminants for THHV_SET_POLICY (mirrors THEMIS_POLICY_* in thhv.h).
#[allow(dead_code)]
pub mod policy_kind {
    pub const CORES: u64 = 0;
    pub const API_MONITOR: u64 = 1;
    pub const DEFAULT_INTR_VISIBILITY: u64 = 2;
    pub const VECTOR_VISIBILITY: u64 = 3;
    pub const VECTOR_REG_READ_SET: u64 = 4;
    pub const VECTOR_REG_WRITE_SET: u64 = 5;
    pub const DEFAULT_EXIT_TRAP: u64 = 6;
    pub const EXIT_REASON_TRAP: u64 = 7;
    pub const EXIT_REASON_REG_READ_SET: u64 = 8;
    pub const EXIT_REASON_REG_WRITE_SET: u64 = 9;
    pub const CPUID_DEFAULT: u64 = 10;
    pub const CPUID_RANGE: u64 = 11;
    pub const CPUID_EMULATE: u64 = 12;
    pub const MSR_DEFAULT: u64 = 13;
    pub const MSR_RANGE: u64 = 14;
    pub const MSR_EMULATE: u64 = 15;
}

/// Standard register set queried/restored by the KVM-like get/set_standard_registers
/// path: 16 GP regs + RIP + RFLAGS (matches `StandardRegisters` layout).
pub(super) const STANDARD_REGS: [VpRegister; 18] = [
    VpRegister::Rax,
    VpRegister::Rbx,
    VpRegister::Rcx,
    VpRegister::Rdx,
    VpRegister::Rsi,
    VpRegister::Rdi,
    VpRegister::Rsp,
    VpRegister::Rbp,
    VpRegister::R8,
    VpRegister::R9,
    VpRegister::R10,
    VpRegister::R11,
    VpRegister::R12,
    VpRegister::R13,
    VpRegister::R14,
    VpRegister::R15,
    VpRegister::Rip,
    VpRegister::Rflags,
];

// Silence dead-code warnings for the full-set re-export until consumers land.
#[allow(dead_code)]
const _: &[VpRegister] = ALL_VP_REGISTERS;

pub(super) const EMPTY_BOOT_MSRS: [MsrEntry; 0] = [];
