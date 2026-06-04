//! Themis backend for cloud-hypervisor's `Hypervisor` / `Vm` / `Vcpu` traits.
//!
//! # Trust model (read this before touching any stub)
//!
//! Themis is a capability-based hypervisor: capavisor (bare metal, L0) is the
//! Trusted Computing Base; cloud-hypervisor (L1, dom0 userspace) is **not
//! trusted**.  Capavisor owns all per-vCPU hardware state (VMCS guest area,
//! FPU, LAPIC, MSRs, MP state) and configures it at VMCS init time in
//! `themis/capavisor/src/arch/x86_64/boot/vmcs.rs`.  cloud-hypervisor reaches
//! capavisor only through the THHV ioctl ABI exposed by `thhv.ko`, which
//! enforces capability checks before any state change.
//!
//! This means many `Vcpu` methods that on KVM/MSHV write directly into the
//! vCPU's hardware state are deliberately **no-ops** here: the corresponding
//! state is set by capavisor, not by CHV.  Each such stub carries a
//! `// Owned by capavisor:` comment.  Conversely, methods that *do* require
//! capavisor cooperation (e.g. `set_sregs`, `set_regs`, `set_cpuid2` for the
//! trap-and-emulate fallback path) go through THHV ioctls.
//!
//! Capabilities not supported at all (live migration: `state`/`set_state`,
//! `get_clock`/`set_clock`, NMI injection: `nmi`) return
//! `Err(... "not supported")` rather than silently succeeding.
//!
//! # Module layout
//!
//! - [`emulator`]        — instruction emulation for trap-and-emulate paths.
//! - [`abi`]             — THHV ioctl numbers + `Thhv*` wire structs.
//! - [`consts`]          — flag/exit-reason/APIC bit definitions, plus
//!                          re-exports of shared types from `themis-abi`.
//! - [`mmap`]             — `MmapRegion` Send/Sync wrapper for shared meta/comm pages.
//! - [`helpers`]          — register-tuple builders and ioctl wrappers.
//! - [`vm_state`]         — deferred-initialization per-VM state.
//! - [`hypervisor_impl`]  — `ThemisHypervisor` and the `Hypervisor` impl.
//! - [`vm_impl`]          — `ThemisVm` and the `vm::Vm` impl.
//! - [`vcpu`]             — `ThemisVcpu` and the `Vcpu` impl.

pub mod emulator;

mod abi;
mod consts;
mod helpers;
mod hypervisor_impl;
mod mmap;
mod vcpu;
mod vm_impl;
mod vm_state;

pub use abi::{shmem_mode, ThemisIrqRoutingEntry, ThemisStandardRegisters, VcpuThemisState};
pub use consts::policy_kind;
pub use hypervisor_impl::ThemisHypervisor;
pub use vcpu::ThemisVcpu;
pub use vm_impl::ThemisVm;
