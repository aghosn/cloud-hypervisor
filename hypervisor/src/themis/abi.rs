//! THHV ioctl-number computation, ioctl wire structs, and `themic`
//! intercept-message structs.
//!
//! These types match the on-the-wire ABI of the THHV kernel module
//! (`thhv/include/uapi/thhv.h`) and the `themic` shared-page protocol used
//! for VM-exit messages.  Sizes and layouts are pinned by `#[repr(C)]`.

use std::mem::size_of;

use libc::c_ulong;
use serde::{Deserialize, Serialize};

use super::consts::THHV_IOCTL_MAGIC;

const fn ioctl_code(dir: c_ulong, ty: u8, nr: u8, size: usize) -> c_ulong {
    (dir << 30) | ((size as c_ulong) << 16) | ((ty as c_ulong) << 8) | nr as c_ulong
}

const fn ioctl_iow<T>(ty: u8, nr: u8) -> c_ulong {
    ioctl_code(1, ty, nr, size_of::<T>())
}

const fn ioctl_iowr<T>(ty: u8, nr: u8) -> c_ulong {
    ioctl_code(3, ty, nr, size_of::<T>())
}

const fn ioctl_io(ty: u8, nr: u8) -> c_ulong {
    ioctl_code(0, ty, nr, 0)
}

pub(super) const THHV_CREATE_PARTITION: c_ulong = ioctl_iowr::<ThhvCreatePartition>(THHV_IOCTL_MAGIC, 0x01);
pub(super) const THHV_QUERY: c_ulong = ioctl_iowr::<ThhvQuery>(THHV_IOCTL_MAGIC, 0x03);
pub(super) const THHV_INITIALIZE_PARTITION: c_ulong = ioctl_io(THHV_IOCTL_MAGIC, 0x10);
pub(super) const THHV_CREATE_VP: c_ulong = ioctl_iowr::<ThhvCreateVp>(THHV_IOCTL_MAGIC, 0x11);
pub(super) const THHV_SET_GUEST_MEMORY: c_ulong = ioctl_iow::<ThhvSetGuestMemory>(THHV_IOCTL_MAGIC, 0x12);
pub(super) const THHV_IRQFD: c_ulong = ioctl_iow::<ThhvIrqfd>(THHV_IOCTL_MAGIC, 0x13);
pub(super) const THHV_IOEVENTFD: c_ulong = ioctl_iow::<ThhvIoeventfd>(THHV_IOCTL_MAGIC, 0x14);
pub(super) const THHV_SEND_SHARED_META: c_ulong = ioctl_iow::<ThhvInitializePartition>(THHV_IOCTL_MAGIC, 0x17);
pub(super) const THHV_RUN_VP: c_ulong = ioctl_iowr::<ThhvRunVp>(THHV_IOCTL_MAGIC, 0x20);
pub(super) const THHV_GET_VP_STATE: c_ulong = ioctl_iowr::<ThhvVpRegisters>(THHV_IOCTL_MAGIC, 0x21);
pub(super) const THHV_SET_VP_STATE: c_ulong = ioctl_iow::<ThhvVpRegisters>(THHV_IOCTL_MAGIC, 0x22);
pub(super) const THHV_INJECT_INTERRUPT: c_ulong = ioctl_iow::<ThhvInjectInterrupt>(THHV_IOCTL_MAGIC, 0x19);
pub(super) const THHV_SET_POLICY: c_ulong = ioctl_iow::<ThhvSetPolicy>(THHV_IOCTL_MAGIC, 0x1a);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ThhvCreatePartition {
    pub cores_mask: u64,
    pub api_flags: u64,
    pub sched_policy: u32,
    pub num_vps: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ThhvCreateVp {
    pub vp_index: u32,
    pub rsvd: u32,
    pub meta_uaddr: u64,
    pub meta_size: u64,
    pub comm_uaddr: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ThhvInitializePartition {
    pub meta_uaddr: u64,
    pub meta_size: u64,
    pub apic_access_uaddr: u64,
    pub apic_access_size: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct ThhvSetGuestMemory {
    pub guest_pfn: u64,
    pub userspace_addr: u64,
    pub size: u64,
    pub flags: u32,
    pub rights: u32,
    pub attrs: u64,
    pub shmem_mode: u32,
    pub shmem_count: u32,
    pub shmem_path: [u8; THHV_SHMEM_PATH_MAX],
}

impl Default for ThhvSetGuestMemory {
    fn default() -> Self {
        Self {
            guest_pfn: 0,
            userspace_addr: 0,
            size: 0,
            flags: 0,
            rights: 0,
            attrs: 0,
            shmem_mode: 0,
            shmem_count: 0,
            shmem_path: [0u8; THHV_SHMEM_PATH_MAX],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ThhvVpRegisters {
    pub count: u32,
    pub rsvd: u32,
    pub regs: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ThhvRegNameValue {
    pub name: u64,
    pub value: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ThhvIrqfd {
    pub fd: i32,
    pub gsi: u32,
    pub flags: u32,
    pub vector: u32,     // MSI vector to inject (0 = use gsi as vector)
    pub vp_index: u32,   // Target VP for injection (0 = BSP)
    pub rsvd: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ThhvIoeventfd {
    pub fd: i32,
    pub flags: u32,
    pub addr: u64,
    pub len: u32,
    pub datamatch: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct ThhvRunVp {
    pub msg_buf: [u8; 256],
}

impl Default for ThhvRunVp {
    fn default() -> Self {
        Self { msg_buf: [0; 256] }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ThhvSetPolicy {
    pub kind: u64,
    pub key: u64,
    pub sub_key: u64,
    pub value: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ThhvInjectInterrupt {
    pub vp_index: u32,
    pub vector: u8,
    pub pad: [u8; 3],
}

pub(super) const THHV_SHMEM_PATH_MAX: usize = 256;

#[allow(dead_code)]
pub mod shmem_mode {
    pub const NONE: u32 = 0;
    pub const ALIAS: u32 = 1;
    pub const CARVE: u32 = 2;
    pub const PLUG: u32 = 3;
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ThhvQuery {
    pub query_type: u32,
    pub rsvd: u32,
    pub result: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ThemicMessageHeader {
    pub message_type: u32,
    pub payload_size: u32,
    pub sequence: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct ThemicInterceptMessage {
    pub header: ThemicMessageHeader,
    pub exit_reason: u32,
    pub instruction_length: u32,
    pub exit_qualification: u64,
    pub guest_physical_address: u64,
    pub guest_rip: u64,
    pub guest_rflags: u64,
    pub port_number: u16,
    pub access_size: u8,
    pub is_write: u8,
    pub reserved: u32,
    pub rax: u64,
    pub instruction_bytes: [u8; 16],
    pub cpuid_rax: u64,
    pub cpuid_rcx: u64,
    pub msr_number: u32,
    pub rsvd2: u32,
    pub msr_value: u64,
}

impl Default for ThemicInterceptMessage {
    fn default() -> Self {
        Self {
            header: ThemicMessageHeader::default(),
            exit_reason: 0,
            instruction_length: 0,
            exit_qualification: 0,
            guest_physical_address: 0,
            guest_rip: 0,
            guest_rflags: 0,
            port_number: 0,
            access_size: 0,
            is_write: 0,
            reserved: 0,
            rax: 0,
            instruction_bytes: [0; 16],
            cpuid_rax: 0,
            cpuid_rcx: 0,
            msr_number: 0,
            rsvd2: 0,
            msr_value: 0,
        }
    }
}

// ── Public state types referenced from `crate::CpuState` / `IrqRoutingEntry` ──

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ThemisStandardRegisters {
    pub rax: u64,
    pub rbx: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rsp: u64,
    pub rbp: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    pub rip: u64,
    pub rflags: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThemisIrqRoutingEntry {
    pub gsi: u32,
    pub msi_address_lo: u32,
    pub msi_address_hi: u32,
    pub msi_data: u32,
    pub irqchip: u32,
    pub pin: u32,
    pub is_msi: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VcpuThemisState {}
