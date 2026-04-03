pub mod emulator;

use std::any::Any;
use std::fs::OpenOptions;
use std::mem::size_of;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::ptr;
use std::sync::{Arc, Mutex};

use anyhow::{Context, anyhow};
use libc::{c_ulong, c_void};
use serde::{Deserialize, Serialize};
use vfio_ioctls::VfioDeviceFd;
use vmm_sys_util::eventfd::EventFd;

use crate::arch::x86::{
    CpuIdEntry, FpuState, LapicState, MsrEntry, SegmentRegister, SpecialRegisters,
};
use crate::cpu::{self, HypervisorCpuError, Vcpu, VmExit};
use crate::hypervisor::{self, Hypervisor};
use crate::vm::{self, DataMatch, InterruptSourceConfig, VmOps};
use crate::{
    CpuState, HypervisorType, HypervisorVmConfig, IoEventAddress, IrqRoutingEntry, MpState,
};

const THHV_IOCTL_MAGIC: u8 = 0xB8;
const THHV_SCHED_SYNC: u32 = 0;
const THHV_MEM_F_UNMAP: u32 = 1 << 0;
const THHV_MEM_F_ALIAS: u32 = 1 << 1;
const THHV_MEM_R_READ: u32 = 1 << 0;
const THHV_MEM_R_WRITE: u32 = 1 << 1;
const THHV_MEM_R_EXEC: u32 = 1 << 2;
const THHV_IRQFD_FLAG_DEASSIGN: u32 = 1 << 0;
const THHV_IOEVENTFD_FLAG_DATAMATCH: u32 = 1 << 0;
const THHV_IOEVENTFD_FLAG_PIO: u32 = 1 << 1;
const THHV_IOEVENTFD_FLAG_DEASSIGN: u32 = 1 << 2;
const THHV_META_PAGES_PER_VP: usize = 3;
const THHV_META_PAGES_SHARED: usize = 4;
const THHV_QUERY_META_PAGES_PER_VP: u32 = 1;
const THHV_QUERY_META_PAGES_SHARED: u32 = 2;

const EXIT_REASON_EXCEPTION_NMI: u32 = 0;
const EXIT_REASON_EXTERNAL_INTERRUPT: u32 = 1;
const EXIT_REASON_TRIPLE_FAULT: u32 = 2;
const EXIT_REASON_CPUID: u32 = 10;
const EXIT_REASON_HLT: u32 = 12;
const EXIT_REASON_VMCALL: u32 = 18;
const EXIT_REASON_CR_ACCESS: u32 = 28;     // Intel SDM: MOV to/from CR0/CR3/CR4/CR8, CLTS, LMSW
const EXIT_REASON_IO_INSTRUCTION: u32 = 30; // Intel SDM: IN, INS, OUT, OUTS
const EXIT_REASON_RDMSR: u32 = 31;
const EXIT_REASON_WRMSR: u32 = 32;
const EXIT_REASON_EPT_VIOLATION: u32 = 48;
const EXIT_REASON_APIC_ACCESS: u32 = 44;  // Intel SDM: APIC-access MMIO trap

const THEMIC_MSG_SHUTDOWN: u32 = 0x0004;

#[allow(dead_code)]
const EPT_VIOLATION_DATA_READ: u64 = 1 << 0;
const EPT_VIOLATION_DATA_WRITE: u64 = 1 << 1;
const EPT_VIOLATION_EXECUTE: u64 = 1 << 2;

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

const THHV_CREATE_PARTITION: c_ulong = ioctl_iowr::<ThhvCreatePartition>(THHV_IOCTL_MAGIC, 0x01);
const THHV_QUERY: c_ulong = ioctl_iowr::<ThhvQuery>(THHV_IOCTL_MAGIC, 0x03);
const THHV_INITIALIZE_PARTITION: c_ulong = ioctl_io(THHV_IOCTL_MAGIC, 0x10);
const THHV_CREATE_VP: c_ulong = ioctl_iowr::<ThhvCreateVp>(THHV_IOCTL_MAGIC, 0x11);
const THHV_SET_GUEST_MEMORY: c_ulong = ioctl_iow::<ThhvSetGuestMemory>(THHV_IOCTL_MAGIC, 0x12);
const THHV_IRQFD: c_ulong = ioctl_iow::<ThhvIrqfd>(THHV_IOCTL_MAGIC, 0x13);
const THHV_IOEVENTFD: c_ulong = ioctl_iow::<ThhvIoeventfd>(THHV_IOCTL_MAGIC, 0x14);
const THHV_SEND_SHARED_META: c_ulong = ioctl_iow::<ThhvInitializePartition>(THHV_IOCTL_MAGIC, 0x17);
const THHV_RUN_VP: c_ulong = ioctl_iowr::<ThhvRunVp>(THHV_IOCTL_MAGIC, 0x20);
const THHV_GET_VP_STATE: c_ulong = ioctl_iowr::<ThhvVpRegisters>(THHV_IOCTL_MAGIC, 0x21);
const THHV_SET_VP_STATE: c_ulong = ioctl_iow::<ThhvVpRegisters>(THHV_IOCTL_MAGIC, 0x22);
const THHV_SET_INTR_POLICY: c_ulong = ioctl_iow::<ThhvSetIntrPolicy>(THHV_IOCTL_MAGIC, 0x18);
const THHV_INJECT_INTERRUPT: c_ulong = ioctl_iow::<ThhvInjectInterrupt>(THHV_IOCTL_MAGIC, 0x19);

const THHV_VP_REG_RAX: u64 = 0x00;
const THHV_VP_REG_RBX: u64 = 0x01;
const THHV_VP_REG_RCX: u64 = 0x02;
const THHV_VP_REG_RDX: u64 = 0x03;
const THHV_VP_REG_RSI: u64 = 0x04;
const THHV_VP_REG_RDI: u64 = 0x05;
const THHV_VP_REG_RBP: u64 = 0x06;
const THHV_VP_REG_R8: u64 = 0x07;
const THHV_VP_REG_R9: u64 = 0x08;
const THHV_VP_REG_R10: u64 = 0x09;
const THHV_VP_REG_R11: u64 = 0x0A;
const THHV_VP_REG_R12: u64 = 0x0B;
const THHV_VP_REG_R13: u64 = 0x0C;
const THHV_VP_REG_R14: u64 = 0x0D;
const THHV_VP_REG_R15: u64 = 0x0E;
const THHV_VP_REG_RSP: u64 = 0x10;
const THHV_VP_REG_RIP: u64 = 0x11;
const THHV_VP_REG_RFLAGS: u64 = 0x12;
const THHV_VP_REG_CR0: u64 = 0x20;
const THHV_VP_REG_CR3: u64 = 0x21;
const THHV_VP_REG_CR4: u64 = 0x22;
const THHV_VP_REG_EFER: u64 = 0x23;
const THHV_VP_REG_CS_SEL: u64 = 0x30;
const THHV_VP_REG_DS_SEL: u64 = 0x31;
const THHV_VP_REG_ES_SEL: u64 = 0x32;
const THHV_VP_REG_FS_SEL: u64 = 0x33;
const THHV_VP_REG_GS_SEL: u64 = 0x34;
const THHV_VP_REG_SS_SEL: u64 = 0x35;
const THHV_VP_REG_TR_SEL: u64 = 0x36;
const THHV_VP_REG_LDTR_SEL: u64 = 0x37;
const THHV_VP_REG_CS_BASE: u64 = 0x40;
const THHV_VP_REG_DS_BASE: u64 = 0x41;
const THHV_VP_REG_ES_BASE: u64 = 0x42;
const THHV_VP_REG_FS_BASE: u64 = 0x43;
const THHV_VP_REG_GS_BASE: u64 = 0x44;
const THHV_VP_REG_SS_BASE: u64 = 0x45;
const THHV_VP_REG_TR_BASE: u64 = 0x46;
const THHV_VP_REG_LDTR_BASE: u64 = 0x47;
const THHV_VP_REG_CS_LIM: u64 = 0x50;
const THHV_VP_REG_DS_LIM: u64 = 0x51;
const THHV_VP_REG_ES_LIM: u64 = 0x52;
const THHV_VP_REG_FS_LIM: u64 = 0x53;
const THHV_VP_REG_GS_LIM: u64 = 0x54;
const THHV_VP_REG_SS_LIM: u64 = 0x55;
const THHV_VP_REG_TR_LIM: u64 = 0x56;
const THHV_VP_REG_LDTR_LIM: u64 = 0x57;
const THHV_VP_REG_CS_AR: u64 = 0x60;
const THHV_VP_REG_DS_AR: u64 = 0x61;
const THHV_VP_REG_ES_AR: u64 = 0x62;
const THHV_VP_REG_FS_AR: u64 = 0x63;
const THHV_VP_REG_GS_AR: u64 = 0x64;
const THHV_VP_REG_SS_AR: u64 = 0x65;
const THHV_VP_REG_TR_AR: u64 = 0x66;
const THHV_VP_REG_LDTR_AR: u64 = 0x67;
const THHV_VP_REG_GDTR_BASE: u64 = 0x70;
const THHV_VP_REG_GDTR_LIM: u64 = 0x71;
const THHV_VP_REG_IDTR_BASE: u64 = 0x72;
const THHV_VP_REG_IDTR_LIM: u64 = 0x73;
const THHV_VP_REG_APIC_BASE: u64 = 0xA0;
const THHV_VP_REG_ACTIVITY_STATE: u64 = 0xB0;

const STANDARD_REG_NAMES: [u64; 18] = [
    THHV_VP_REG_RAX,
    THHV_VP_REG_RBX,
    THHV_VP_REG_RCX,
    THHV_VP_REG_RDX,
    THHV_VP_REG_RSI,
    THHV_VP_REG_RDI,
    THHV_VP_REG_RSP,
    THHV_VP_REG_RBP,
    THHV_VP_REG_R8,
    THHV_VP_REG_R9,
    THHV_VP_REG_R10,
    THHV_VP_REG_R11,
    THHV_VP_REG_R12,
    THHV_VP_REG_R13,
    THHV_VP_REG_R14,
    THHV_VP_REG_R15,
    THHV_VP_REG_RIP,
    THHV_VP_REG_RFLAGS,
];

const EMPTY_BOOT_MSRS: [MsrEntry; 0] = [];

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ThhvCreatePartition {
    cores_mask: u64,
    api_flags: u64,
    sched_policy: u32,
    num_vps: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ThhvCreateVp {
    vp_index: u32,
    rsvd: u32,
    meta_uaddr: u64,
    meta_size: u64,
    comm_uaddr: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ThhvInitializePartition {
    meta_uaddr: u64,
    meta_size: u64,
    apic_access_uaddr: u64,
    apic_access_size: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ThhvSetGuestMemory {
    guest_pfn: u64,
    userspace_addr: u64,
    size: u64,
    flags: u32,
    rights: u32,
    attrs: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ThhvVpRegisters {
    count: u32,
    rsvd: u32,
    regs: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ThhvRegNameValue {
    name: u64,
    value: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ThhvIrqfd {
    fd: i32,
    gsi: u32,
    flags: u32,
    vector: u32,     // MSI vector to inject (0 = use gsi as vector)
    vp_index: u32,   // Target VP for injection (0 = BSP)
    rsvd: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ThhvIoeventfd {
    fd: i32,
    flags: u32,
    addr: u64,
    len: u32,
    datamatch: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ThhvRunVp {
    msg_buf: [u8; 256],
}

impl Default for ThhvRunVp {
    fn default() -> Self {
        Self { msg_buf: [0; 256] }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ThhvSetIntrPolicy {
    vector: u8,
    visibility: u8,
    pad: [u8; 6],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ThhvInjectInterrupt {
    vp_index: u32,
    vector: u8,
    pad: [u8; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ThhvQuery {
    query_type: u32,
    rsvd: u32,
    result: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ThemicMessageHeader {
    message_type: u32,
    payload_size: u32,
    sequence: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ThemicInterceptMessage {
    header: ThemicMessageHeader,
    exit_reason: u32,
    instruction_length: u32,
    exit_qualification: u64,
    guest_physical_address: u64,
    guest_rip: u64,
    guest_rflags: u64,
    port_number: u16,
    access_size: u8,
    is_write: u8,
    reserved: u32,
    rax: u64,
    instruction_bytes: [u8; 16],
    cpuid_rax: u64,
    cpuid_rcx: u64,
    msr_number: u32,
    rsvd2: u32,
    msr_value: u64,
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

struct MmapRegion {
    addr: *mut c_void,
    len: usize,
}

// SAFETY: The mapping lifetime is owned by the struct and access is coordinated by the VMM.
unsafe impl Send for MmapRegion {}
// SAFETY: The mapping points to shared memory supplied to the kernel; raw pointer sharing is intentional.
unsafe impl Sync for MmapRegion {}

impl MmapRegion {
    fn new_shared_anonymous(len: usize) -> anyhow::Result<Self> {
        // SAFETY: Valid mmap arguments for a shared anonymous region.
        let addr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }

        Ok(Self { addr, len })
    }

    fn as_u64(&self) -> u64 {
        self.addr as usize as u64
    }
}

impl Drop for MmapRegion {
    fn drop(&mut self) {
        if !self.addr.is_null() && self.addr != libc::MAP_FAILED {
            // SAFETY: Region was created by mmap with this address/length.
            unsafe {
                libc::munmap(self.addr, self.len);
            }
        }
    }
}

struct ThemisVmState {
    fd: Arc<OwnedFd>,
    initialized: Mutex<bool>,
    page_size: usize,
    vp_meta_pages: usize,
    _shared_meta: MmapRegion,
    /// APIC-access sentinel page mapped at GPA 0xFEE00000 in the child domain.
    /// Kept alive here so the pinned page is not freed while the domain runs.
    _apic_access: MmapRegion,
    /// EventFd for the periodic timer interrupt injection.
    /// Kept alive so the IRQFd registration in thhv.ko remains valid.
    _timer_eventfd: Mutex<Option<EventFd>>,
    /// Guest memory regions recorded during create_user_memory_region but not
    /// yet sent to the domain.  THHV_SET_GUEST_MEMORY is deferred until
    /// ensure_initialized() so that cloud-hypervisor can finish writing
    /// firmware/ACPI/kernel data into the mmap'd pages while dom0 still holds
    /// EPT access.  Once SET_GUEST_MEMORY fires the capavisor removes those
    /// pages from dom0's EPT, so all writes must complete before then.
    pending_memory: Mutex<Vec<ThhvSetGuestMemory>>,
    /// GSI → MSI vector mapping.  Updated by set_gsi_routing() when the guest
    /// configures MSI Address/Data.  Used by register_irqfd() to pass the
    /// correct injection vector to thhv (instead of the GSI number).
    gsi_vectors: Mutex<std::collections::HashMap<u32, u8>>,
    /// Per-vCPU file descriptors (raw), indexed by vCPU ID.  Used for
    /// cross-vCPU state updates (e.g., BSP sending SIPI to an AP).
    vp_fds: Mutex<Vec<RawFd>>,
    /// Per-vCPU software LAPIC register state (1024 × u32 = 4 KiB per vCPU).
    /// Used when VIRTUALIZE_APIC_ACCESSES is not available: LAPIC MMIO
    /// accesses cause EPT violations that are emulated here in software.
    lapic_regs: Mutex<Vec<[u32; 1024]>>,
}

impl ThemisVmState {
    fn ensure_initialized(&self) -> anyhow::Result<()> {
        let mut initialized = self.initialized.lock().unwrap();
        if *initialized {
            return Ok(());
        }

        // Flush all deferred SET_GUEST_MEMORY calls before sealing.
        // After this point dom0 loses EPT access to dom1's guest pages.
        let regions: Vec<ThhvSetGuestMemory> =
            self.pending_memory.lock().unwrap().drain(..).collect();
        eprintln!("[THEMIS-DBG] ensure_initialized: flushing {} memory regions", regions.len());
        for (i, mut region) in regions.into_iter().enumerate() {
            eprintln!("[THEMIS-DBG] SET_GUEST_MEMORY[{i}] gfn=0x{:x} size=0x{:x}", region.guest_pfn, region.size);
            ioctl_with_mut_ref(self.fd.as_raw_fd(), THHV_SET_GUEST_MEMORY, &mut region)
                .context("deferred THHV_SET_GUEST_MEMORY failed")?;
            eprintln!("[THEMIS-DBG] SET_GUEST_MEMORY[{i}] done");
        }

        eprintln!("[THEMIS-DBG] INITIALIZE_PARTITION (seal)...");
        ioctl_noarg(self.fd.as_raw_fd(), THHV_INITIALIZE_PARTITION)
            .context("failed to initialize Themis partition")?;
        eprintln!("[THEMIS-DBG] INITIALIZE_PARTITION done");
        *initialized = true;

        // LAPIC timer emulation is set up per-vCPU in create_vcpu():
        // timerfd + irqfd + background thread.  WRMSR 0x6E0 exits
        // (trapped by the child MSR bitmap in capavisor) are handled
        // in handle_wrmsr_exit() which reprograms the timerfd.

        Ok(())
    }
}

pub struct ThemisHypervisor {
    fd: OwnedFd,
    page_size: usize,
    vp_meta_pages: usize,
    shared_meta_pages: usize,
}

pub struct ThemisVm {
    state: Arc<ThemisVmState>,
}

pub struct ThemisVcpu {
    fd: OwnedFd,
    _vp_index: u32,
    vm_state: Arc<ThemisVmState>,
    vm_ops: Option<Arc<dyn VmOps>>,
    _meta: MmapRegion,
    _comm: MmapRegion,
    /// CPUID policy set by CHV via set_cpuid2() before the first run.
    /// Searched by (function, index) on every CPUID exit.
    cpuid: std::sync::Mutex<Vec<CpuIdEntry>>,
    /// LAPIC timer emulation: timerfd armed to the TSC deadline.
    /// A background thread reads the timerfd and signals `timer_irq`.
    timer_fd: OwnedFd,
    /// EventFd registered as irqfd for LOCAL_TIMER_VECTOR (0xEC).
    /// Signalled by the timer thread when the deadline fires.
    _timer_irq: EventFd,
}

impl ThemisVcpu {
    /// Access the VmOps callbacks (MMIO/PIO/guest-mem dispatch).
    pub fn vm_ops(&self) -> Option<&Arc<dyn VmOps>> {
        self.vm_ops.as_ref()
    }
}

impl ThemisHypervisor {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> hypervisor::Result<Arc<dyn Hypervisor>> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_CLOEXEC)
            .open("/dev/thhv")
            .map_err(|e| hypervisor::HypervisorError::HypervisorCreate(e.into()))?;

        let page_size =
            page_size().map_err(|e| hypervisor::HypervisorError::HypervisorCreate(e.into()))?;
        let fd: OwnedFd = file.into();
        let vp_meta_pages = query_meta_pages(fd.as_raw_fd(), THHV_QUERY_META_PAGES_PER_VP)
            .unwrap_or(THHV_META_PAGES_PER_VP as u64) as usize;
        let shared_meta_pages = query_meta_pages(fd.as_raw_fd(), THHV_QUERY_META_PAGES_SHARED)
            .unwrap_or(THHV_META_PAGES_SHARED as u64) as usize;

        Ok(Arc::new(Self {
            fd,
            page_size,
            vp_meta_pages,
            shared_meta_pages,
        }))
    }

    pub fn is_available() -> hypervisor::Result<bool> {
        Ok(Path::new("/dev/thhv").exists())
    }
}

impl Hypervisor for ThemisHypervisor {
    fn hypervisor_type(&self) -> HypervisorType {
        HypervisorType::Themis
    }

    fn create_vm(&self, _config: HypervisorVmConfig) -> hypervisor::Result<Arc<dyn vm::Vm>> {
        let mut create = ThhvCreatePartition {
            cores_mask: !0,
            api_flags: !0,
            sched_policy: THHV_SCHED_SYNC,
            num_vps: self.get_max_vcpus(),
        };

        let part_fd =
            ioctl_with_mut_ref_ret_fd(self.fd.as_raw_fd(), THHV_CREATE_PARTITION, &mut create)
                .map_err(|e| hypervisor::HypervisorError::VmCreate(e.into()))?;

        let shared_meta = MmapRegion::new_shared_anonymous(self.shared_meta_pages * self.page_size)
            .map_err(|e| hypervisor::HypervisorError::VmCreate(e.into()))?;
        let apic_access = MmapRegion::new_shared_anonymous(self.page_size)
            .map_err(|e| hypervisor::HypervisorError::VmCreate(e.into()))?;
        let mut init = ThhvInitializePartition {
            meta_uaddr: shared_meta.as_u64(),
            meta_size: (self.shared_meta_pages * self.page_size) as u64,
            apic_access_uaddr: apic_access.as_u64(),
            apic_access_size: self.page_size as u64,
        };
        ioctl_with_mut_ref(part_fd.as_raw_fd(), THHV_SEND_SHARED_META, &mut init)
            .map_err(|e| hypervisor::HypervisorError::VmCreate(e.into()))?;
        // THHV_INITIALIZE_PARTITION (seal) is deferred until first run() via
        // ensure_initialized().  This allows per-VP META pages to be sent via
        // THHV_CREATE_VP while the domain is still unsealed, so the capavisor
        // processes them as immediate GiveMetaMem updates rather than pending
        // capabilities that would require a later ACCEPT before ADD_VP.

        Ok(Arc::new(ThemisVm {
            state: Arc::new(ThemisVmState {
                fd: Arc::new(part_fd),
                initialized: Mutex::new(false),
                page_size: self.page_size,
                vp_meta_pages: self.vp_meta_pages,
                _shared_meta: shared_meta,
                _apic_access: apic_access,
                _timer_eventfd: Mutex::new(None),
                pending_memory: Mutex::new(Vec::new()),
                gsi_vectors: Mutex::new(std::collections::HashMap::new()),
                vp_fds: Mutex::new(Vec::new()),
                lapic_regs: Mutex::new(Vec::new()),
            }),
        }))
    }

    fn get_supported_cpuid(&self) -> hypervisor::Result<Vec<CpuIdEntry>> {
        // Read the host CPUID directly. Cloud-hypervisor will patch the result
        // (hypervisor bit, APIC IDs, topology, etc.) before passing it to the guest.
        // We enumerate standard leaves 0..=max and extended leaves 0x8000_0000..=max.
        let mut entries: Vec<CpuIdEntry> = Vec::new();

        let collect = |entries: &mut Vec<CpuIdEntry>, leaf: u32, max_leaf: u32| {
            for function in leaf..=max_leaf {
                // Most leaves only have index 0; subleaf enumeration is handled
                // by generate_common_cpuid for the leaves it cares about (0x4, 0x7, 0xb, etc.).
                for index in 0u32..=3 {
                    let result = unsafe {
                        std::arch::x86_64::__cpuid_count(function, index)
                    };
                    // Skip entirely-zero sub-leaves (not present).
                    if result.eax == 0 && result.ebx == 0 && result.ecx == 0 && result.edx == 0 {
                        if index > 0 { break; }
                    }
                    entries.push(CpuIdEntry {
                        function,
                        index,
                        flags: 0,
                        eax: result.eax,
                        ebx: result.ebx,
                        ecx: result.ecx,
                        edx: result.edx,
                    });
                }
            }
        };

        let max_leaf = unsafe { std::arch::x86_64::__cpuid(0).eax };
        collect(&mut entries, 0, max_leaf.min(0x20));

        let max_ext = unsafe { std::arch::x86_64::__cpuid(0x8000_0000).eax };
        if max_ext >= 0x8000_0000 {
            collect(&mut entries, 0x8000_0000, max_ext.min(0x8000_0020));
        }

        Ok(entries)
    }

    fn get_max_vcpus(&self) -> u32 {
        256
    }

    fn get_guest_debug_hw_bps(&self) -> usize {
        0
    }
}

impl vm::Vm for ThemisVm {
    fn set_identity_map_address(&self, _address: u64) -> vm::Result<()> {
        Ok(())
    }

    fn set_tss_address(&self, _offset: usize) -> vm::Result<()> {
        Ok(())
    }

    fn create_irq_chip(&self) -> vm::Result<()> {
        Ok(())
    }

    fn register_irqfd(&self, fd: &EventFd, gsi: u32) -> vm::Result<()> {
        let vector = self.state.gsi_vectors.lock().unwrap()
            .get(&gsi).copied().unwrap_or(0) as u32;
        eprintln!("[THEMIS-DBG] register_irqfd gsi={gsi} vec={vector} fd={}", fd.as_raw_fd());
        let mut irqfd = ThhvIrqfd {
            fd: fd.as_raw_fd(),
            gsi,
            flags: 0,
            vector,
            vp_index: 0,
            rsvd: 0,
        };
        let r = ioctl_with_mut_ref(self.state.fd.as_raw_fd(), THHV_IRQFD, &mut irqfd)
            .map_err(|e| vm::HypervisorVmError::RegisterIrqFd(e.into()));
        eprintln!("[THEMIS-DBG] register_irqfd gsi={gsi} vec={vector} result={r:?}");
        r
    }

    fn unregister_irqfd(&self, fd: &EventFd, gsi: u32) -> vm::Result<()> {
        let mut irqfd = ThhvIrqfd {
            fd: fd.as_raw_fd(),
            gsi,
            flags: THHV_IRQFD_FLAG_DEASSIGN,
            vector: 0,
            vp_index: 0,
            rsvd: 0,
        };
        ioctl_with_mut_ref(self.state.fd.as_raw_fd(), THHV_IRQFD, &mut irqfd)
            .map_err(|e| vm::HypervisorVmError::UnregisterIrqFd(e.into()))
    }

    fn create_vcpu(&self, id: u32, vm_ops: Option<Arc<dyn VmOps>>) -> vm::Result<Box<dyn Vcpu>> {
        let meta =
            MmapRegion::new_shared_anonymous(self.state.vp_meta_pages * self.state.page_size)
                .map_err(|e| vm::HypervisorVmError::CreateVcpu(e.into()))?;
        let comm = MmapRegion::new_shared_anonymous(self.state.page_size)
            .map_err(|e| vm::HypervisorVmError::CreateVcpu(e.into()))?;
        let mut create = ThhvCreateVp {
            vp_index: id,
            rsvd: 0,
            meta_uaddr: meta.as_u64(),
            meta_size: (self.state.vp_meta_pages * self.state.page_size) as u64,
            comm_uaddr: comm.as_u64(),
        };
        let vcpu_fd =
            ioctl_with_mut_ref_ret_fd(self.state.fd.as_raw_fd(), THHV_CREATE_VP, &mut create)
                .map_err(|e| vm::HypervisorVmError::CreateVcpu(e.into()))?;

        // ── LAPIC timer emulation ──
        // Create a timerfd that the vCPU run-loop reprograms on each
        // WRMSR IA32_TSC_DEADLINE (0x6E0).  A background thread bridges
        // timerfd expirations to an irqfd → thhv injects vector 0xEC.
        const LOCAL_TIMER_VECTOR: u32 = 0xEC;
        let timer_fd = unsafe {
            let fd = libc::timerfd_create(libc::CLOCK_MONOTONIC, libc::TFD_CLOEXEC);
            if fd < 0 {
                return Err(vm::HypervisorVmError::CreateVcpu(
                    std::io::Error::last_os_error().into(),
                ));
            }
            OwnedFd::from_raw_fd(fd)
        };

        let timer_irq = EventFd::new(libc::EFD_CLOEXEC)
            .map_err(|e| vm::HypervisorVmError::CreateVcpu(e.into()))?;

        // Register the EventFd as an irqfd for vector 0xEC, targeting this vCPU.
        {
            let mut irqfd = ThhvIrqfd {
                fd: timer_irq.as_raw_fd(),
                gsi: LOCAL_TIMER_VECTOR,
                flags: 0,
                vector: LOCAL_TIMER_VECTOR,
                vp_index: id as u32,
                rsvd: 0,
            };
            ioctl_with_mut_ref(self.state.fd.as_raw_fd(), THHV_IRQFD, &mut irqfd)
                .map_err(|e| vm::HypervisorVmError::CreateVcpu(e.into()))?;
        }

        // Spawn a thread that waits on timerfd and signals the irqfd.
        let irq_clone = timer_irq.try_clone().unwrap();
        let timer_fd_dup = unsafe {
            let raw = libc::dup(timer_fd.as_raw_fd());
            assert!(raw >= 0, "dup(timer_fd) failed");
            OwnedFd::from_raw_fd(raw)
        };
        std::thread::Builder::new()
            .name(format!("lapic-timer-{id}"))
            .spawn(move || {
                let raw = timer_fd_dup.as_raw_fd();
                let mut buf = [0u8; 8];
                loop {
                    let n = unsafe { libc::read(raw, buf.as_mut_ptr() as *mut c_void, 8) };
                    if n <= 0 {
                        break;
                    }
                    let _ = irq_clone.write(1);
                }
                drop(timer_fd_dup);
            })
            .map_err(|e| vm::HypervisorVmError::CreateVcpu(e.into()))?;

        eprintln!(
            "[THEMIS-DBG] vCPU {id}: LAPIC timer emulation ready (timerfd={}, irqfd vec=0x{LOCAL_TIMER_VECTOR:X})",
            timer_fd.as_raw_fd()
        );

        // Register VP fd for cross-vCPU state updates (SIPI delivery).
        {
            let raw_fd = vcpu_fd.as_raw_fd();
            let mut fds = self.state.vp_fds.lock().unwrap();
            // Grow to fit this vCPU ID.
            while fds.len() <= id as usize {
                fds.push(-1);
            }
            fds[id as usize] = raw_fd;
        }

        // Non-BSP vCPUs start in wait-for-SIPI state.  The thhv driver
        // blocks VP_RUN until ACTIVITY_STATE is set back to 0 (by the
        // BSP's SIPI handler in handle_run_exit).
        if id > 0 {
            let regs = [reg(THHV_VP_REG_ACTIVITY_STATE, 3)];
            let mut regs = regs.to_vec();
            let mut header = ThhvVpRegisters {
                count: regs.len() as u32,
                rsvd: 0,
                regs: regs.as_mut_ptr() as usize as u64,
            };
            ioctl_with_mut_ref(vcpu_fd.as_raw_fd(), THHV_SET_VP_STATE, &mut header)
                .map_err(|e| vm::HypervisorVmError::CreateVcpu(e.into()))?;
            eprintln!("[THEMIS-DBG] vCPU {id}: set ACTIVITY_STATE=3 (wait-for-SIPI)");
        }

        // Initialize per-vCPU software LAPIC state.
        {
            let mut lapic = self.state.lapic_regs.lock().unwrap();
            while lapic.len() <= id as usize {
                lapic.push([0u32; 1024]);
            }
            let regs = &mut lapic[id as usize];
            regs[0x020 / 4] = (id as u32) << 24;        // APIC ID
            regs[0x030 / 4] = 0x0005_0014;               // Version: v20, 6 LVT entries
            regs[0x0D0 / 4] = 0;                          // LDR
            regs[0x0E0 / 4] = 0xFFFF_FFFF;                // DFR: flat model
            regs[0x0F0 / 4] = 0x0000_01FF;                // SVR: APIC enabled, vector 0xFF
            // LVT entries: masked by default
            regs[0x320 / 4] = 0x0001_0000;                // LVT Timer (masked)
            regs[0x350 / 4] = 0x0001_0000;                // LVT LINT0 (masked)
            regs[0x360 / 4] = 0x0001_0000;                // LVT LINT1 (masked)
            regs[0x370 / 4] = 0x0001_0000;                // LVT Error (masked)
            regs[0x340 / 4] = 0x0001_0000;                // LVT PerfMon (masked)
            regs[0x330 / 4] = 0x0001_0000;                // LVT Thermal (masked)
            regs[0x3E0 / 4] = 0x0000_000B;                // Timer DCR: divide by 1
        }

        Ok(Box::new(ThemisVcpu {
            fd: vcpu_fd,
            _vp_index: id,
            vm_state: self.state.clone(),
            vm_ops,
            _meta: meta,
            _comm: comm,
            cpuid: std::sync::Mutex::new(Vec::new()),
            timer_fd,
            _timer_irq: timer_irq,
        }))
    }

    fn register_ioevent(
        &self,
        fd: &EventFd,
        addr: &IoEventAddress,
        datamatch: Option<DataMatch>,
    ) -> vm::Result<()> {
        let (addr, len, mut flags) = match addr {
            IoEventAddress::Pio(port) => (*port, 0, THHV_IOEVENTFD_FLAG_PIO),
            IoEventAddress::Mmio(gpa) => (*gpa, 0, 0),
        };
        let datamatch = match datamatch {
            Some(DataMatch::DataMatch32(v)) => {
                flags |= THHV_IOEVENTFD_FLAG_DATAMATCH;
                v
            }
            Some(DataMatch::DataMatch64(v)) if u32::try_from(v).is_ok() => {
                flags |= THHV_IOEVENTFD_FLAG_DATAMATCH;
                v as u32
            }
            Some(DataMatch::DataMatch64(_)) => {
                return Err(vm::HypervisorVmError::RegisterIoEvent(anyhow!(
                    "64-bit ioeventfd datamatch is not supported by thhv"
                )));
            }
            None => 0,
        };
        let mut ioevent = ThhvIoeventfd {
            fd: fd.as_raw_fd(),
            flags,
            addr,
            len,
            datamatch,
        };
        ioctl_with_mut_ref(self.state.fd.as_raw_fd(), THHV_IOEVENTFD, &mut ioevent)
            .map_err(|e| vm::HypervisorVmError::RegisterIoEvent(e.into()))
    }

    fn unregister_ioevent(&self, fd: &EventFd, addr: &IoEventAddress) -> vm::Result<()> {
        let (addr, len, flags) = match addr {
            IoEventAddress::Pio(port) => (
                *port,
                0,
                THHV_IOEVENTFD_FLAG_PIO | THHV_IOEVENTFD_FLAG_DEASSIGN,
            ),
            IoEventAddress::Mmio(gpa) => (*gpa, 0, THHV_IOEVENTFD_FLAG_DEASSIGN),
        };
        let mut ioevent = ThhvIoeventfd {
            fd: fd.as_raw_fd(),
            flags,
            addr,
            len,
            datamatch: 0,
        };
        ioctl_with_mut_ref(self.state.fd.as_raw_fd(), THHV_IOEVENTFD, &mut ioevent)
            .map_err(|e| vm::HypervisorVmError::UnregisterIoEvent(e.into()))
    }

    fn make_routing_entry(&self, gsi: u32, config: &InterruptSourceConfig) -> IrqRoutingEntry {
        let entry = match config {
            InterruptSourceConfig::LegacyIrq(cfg) => ThemisIrqRoutingEntry {
                gsi,
                msi_address_lo: 0,
                msi_address_hi: 0,
                msi_data: 0,
                irqchip: cfg.irqchip,
                pin: cfg.pin,
                is_msi: false,
            },
            InterruptSourceConfig::MsiIrq(cfg) => ThemisIrqRoutingEntry {
                gsi,
                msi_address_lo: cfg.low_addr,
                msi_address_hi: cfg.high_addr,
                msi_data: cfg.data,
                irqchip: 0,
                pin: 0,
                is_msi: true,
            },
        };
        IrqRoutingEntry::Themis(entry)
    }

    fn set_gsi_routing(&self, entries: &[IrqRoutingEntry]) -> vm::Result<()> {
        let mut map = self.state.gsi_vectors.lock().unwrap();
        map.clear();
        for entry in entries {
            if let IrqRoutingEntry::Themis(e) = entry {
                if e.is_msi {
                    let vector = (e.msi_data & 0xFF) as u8;
                    eprintln!("[THEMIS-DBG] gsi_routing: gsi={} → msi_vector={}", e.gsi, vector);
                    map.insert(e.gsi, vector);
                }
            }
        }
        Ok(())
    }

    unsafe fn create_user_memory_region(
        &self,
        _slot: u32,
        guest_phys_addr: u64,
        memory_size: usize,
        userspace_addr: *mut u8,
        _readonly: bool,
        _log_dirty_pages: bool,
    ) -> vm::Result<()> {
        let region = ThhvSetGuestMemory {
            guest_pfn: guest_phys_addr >> 12,
            userspace_addr: userspace_addr as usize as u64,
            size: memory_size as u64,
            flags: THHV_MEM_F_ALIAS,
            rights: THHV_MEM_R_READ | THHV_MEM_R_WRITE | THHV_MEM_R_EXEC,
            attrs: 0,
        };
        // Defer THHV_SET_GUEST_MEMORY until ensure_initialized() (just before
        // first run()).  Cloud-hypervisor must finish all writes into dom1's
        // guest memory (firmware, ACPI tables, etc.) before we hand those pages
        // to the capavisor; the capavisor removes them from dom0's EPT on SEND.
        self.state.pending_memory.lock().unwrap().push(region);
        Ok(())
    }

    unsafe fn remove_user_memory_region(
        &self,
        _slot: u32,
        guest_phys_addr: u64,
        memory_size: usize,
        userspace_addr: *mut u8,
        _readonly: bool,
        _log_dirty_pages: bool,
    ) -> vm::Result<()> {
        let mut region = ThhvSetGuestMemory {
            guest_pfn: guest_phys_addr >> 12,
            userspace_addr: userspace_addr as usize as u64,
            size: memory_size as u64,
            flags: THHV_MEM_F_UNMAP,
            rights: THHV_MEM_R_READ | THHV_MEM_R_WRITE | THHV_MEM_R_EXEC,
            attrs: 0,
        };
        ioctl_with_mut_ref(
            self.state.fd.as_raw_fd(),
            THHV_SET_GUEST_MEMORY,
            &mut region,
        )
        .map_err(|e| vm::HypervisorVmError::RemoveUserMemory(e.into()))
    }

    fn enable_split_irq(&self) -> vm::Result<()> {
        Ok(())
    }

    fn get_clock(&self) -> vm::Result<crate::ClockData> {
        Err(vm::HypervisorVmError::GetClock(anyhow!("not supported")))
    }

    fn set_clock(&self, _data: &crate::ClockData) -> vm::Result<()> {
        Err(vm::HypervisorVmError::SetClock(anyhow!("not supported")))
    }

    fn create_passthrough_device(&self) -> vm::Result<VfioDeviceFd> {
        Err(vm::HypervisorVmError::CreatePassthroughDevice(anyhow!(
            "not supported"
        )))
    }

    fn start_dirty_log(&self) -> vm::Result<()> {
        Err(vm::HypervisorVmError::StartDirtyLog(anyhow!(
            "not supported"
        )))
    }

    fn stop_dirty_log(&self) -> vm::Result<()> {
        Err(vm::HypervisorVmError::StopDirtyLog(anyhow!(
            "not supported"
        )))
    }

    fn get_dirty_log(&self, _slot: u32, _base_gpa: u64, _memory_size: u64) -> vm::Result<Vec<u64>> {
        Err(vm::HypervisorVmError::GetDirtyLog(anyhow!("not supported")))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Vcpu for ThemisVcpu {
    fn create_standard_regs(&self) -> crate::StandardRegisters {
        crate::StandardRegisters::Themis(ThemisStandardRegisters::default())
    }

    fn get_regs(&self) -> cpu::Result<crate::StandardRegisters> {
        let values = self.get_reg_values(&STANDARD_REG_NAMES)?;
        Ok(crate::StandardRegisters::Themis(ThemisStandardRegisters {
            rax: values[0],
            rbx: values[1],
            rcx: values[2],
            rdx: values[3],
            rsi: values[4],
            rdi: values[5],
            rsp: values[6],
            rbp: values[7],
            r8: values[8],
            r9: values[9],
            r10: values[10],
            r11: values[11],
            r12: values[12],
            r13: values[13],
            r14: values[14],
            r15: values[15],
            rip: values[16],
            rflags: values[17],
        }))
    }

    fn set_regs(&self, regs: &crate::StandardRegisters) -> cpu::Result<()> {
        let regs = match regs {
            crate::StandardRegisters::Themis(regs) => regs,
            #[allow(unreachable_patterns)]
            _ => {
                return Err(HypervisorCpuError::SetStandardRegs(anyhow!(
                    "invalid register backend"
                )));
            }
        };

        let regs = [
            reg(THHV_VP_REG_RAX, regs.rax),
            reg(THHV_VP_REG_RBX, regs.rbx),
            reg(THHV_VP_REG_RCX, regs.rcx),
            reg(THHV_VP_REG_RDX, regs.rdx),
            reg(THHV_VP_REG_RSI, regs.rsi),
            reg(THHV_VP_REG_RDI, regs.rdi),
            reg(THHV_VP_REG_RSP, regs.rsp),
            reg(THHV_VP_REG_RBP, regs.rbp),
            reg(THHV_VP_REG_R8, regs.r8),
            reg(THHV_VP_REG_R9, regs.r9),
            reg(THHV_VP_REG_R10, regs.r10),
            reg(THHV_VP_REG_R11, regs.r11),
            reg(THHV_VP_REG_R12, regs.r12),
            reg(THHV_VP_REG_R13, regs.r13),
            reg(THHV_VP_REG_R14, regs.r14),
            reg(THHV_VP_REG_R15, regs.r15),
            reg(THHV_VP_REG_RIP, regs.rip),
            reg(THHV_VP_REG_RFLAGS, regs.rflags),
        ];
        let result = self.set_reg_values(&regs)
            .map_err(|e| HypervisorCpuError::SetStandardRegs(anyhow!(e.to_string())));
        if let Err(ref e) = result {
            eprintln!("[THEMIS-DBG] set_regs: THHV_SET_VP_STATE FAILED: {e}");
        }
        result
    }

    fn get_sregs(&self) -> cpu::Result<SpecialRegisters> {
        let names = [
            THHV_VP_REG_CR0,
            THHV_VP_REG_CR3,
            THHV_VP_REG_CR4,
            THHV_VP_REG_EFER,
            THHV_VP_REG_APIC_BASE,
            THHV_VP_REG_CS_SEL,
            THHV_VP_REG_CS_BASE,
            THHV_VP_REG_CS_LIM,
            THHV_VP_REG_CS_AR,
            THHV_VP_REG_DS_SEL,
            THHV_VP_REG_DS_BASE,
            THHV_VP_REG_DS_LIM,
            THHV_VP_REG_DS_AR,
            THHV_VP_REG_ES_SEL,
            THHV_VP_REG_ES_BASE,
            THHV_VP_REG_ES_LIM,
            THHV_VP_REG_ES_AR,
            THHV_VP_REG_FS_SEL,
            THHV_VP_REG_FS_BASE,
            THHV_VP_REG_FS_LIM,
            THHV_VP_REG_FS_AR,
            THHV_VP_REG_GS_SEL,
            THHV_VP_REG_GS_BASE,
            THHV_VP_REG_GS_LIM,
            THHV_VP_REG_GS_AR,
            THHV_VP_REG_SS_SEL,
            THHV_VP_REG_SS_BASE,
            THHV_VP_REG_SS_LIM,
            THHV_VP_REG_SS_AR,
            THHV_VP_REG_TR_SEL,
            THHV_VP_REG_TR_BASE,
            THHV_VP_REG_TR_LIM,
            THHV_VP_REG_TR_AR,
            THHV_VP_REG_LDTR_SEL,
            THHV_VP_REG_LDTR_BASE,
            THHV_VP_REG_LDTR_LIM,
            THHV_VP_REG_LDTR_AR,
            THHV_VP_REG_GDTR_BASE,
            THHV_VP_REG_GDTR_LIM,
            THHV_VP_REG_IDTR_BASE,
            THHV_VP_REG_IDTR_LIM,
        ];
        let values = self.get_reg_values(&names)?;

        Ok(SpecialRegisters {
            cr0: values[0],
            cr2: 0,
            cr3: values[1],
            cr4: values[2],
            cr8: 0,
            efer: values[3],
            apic_base: values[4],
            cs: segment_from_raw(values[5], values[6], values[7], values[8]),
            ds: segment_from_raw(values[9], values[10], values[11], values[12]),
            es: segment_from_raw(values[13], values[14], values[15], values[16]),
            fs: segment_from_raw(values[17], values[18], values[19], values[20]),
            gs: segment_from_raw(values[21], values[22], values[23], values[24]),
            ss: segment_from_raw(values[25], values[26], values[27], values[28]),
            tr: segment_from_raw(values[29], values[30], values[31], values[32]),
            ldt: segment_from_raw(values[33], values[34], values[35], values[36]),
            gdt: crate::arch::x86::DescriptorTable {
                base: values[37],
                limit: values[38] as u16,
            },
            idt: crate::arch::x86::DescriptorTable {
                base: values[39],
                limit: values[40] as u16,
            },
            interrupt_bitmap: [0; 4],
        })
    }

    fn set_sregs(&self, sregs: &SpecialRegisters) -> cpu::Result<()> {
        let mut regs = Vec::new();
        regs.extend_from_slice(&[
            reg(THHV_VP_REG_CR0, sregs.cr0),
            reg(THHV_VP_REG_CR3, sregs.cr3),
            reg(THHV_VP_REG_CR4, sregs.cr4),
            reg(THHV_VP_REG_EFER, sregs.efer),
            reg(THHV_VP_REG_APIC_BASE, sregs.apic_base),
        ]);
        regs.extend_from_slice(&segment_regs(
            THHV_VP_REG_CS_SEL,
            THHV_VP_REG_CS_BASE,
            THHV_VP_REG_CS_LIM,
            THHV_VP_REG_CS_AR,
            &sregs.cs,
        ));
        regs.extend_from_slice(&segment_regs(
            THHV_VP_REG_DS_SEL,
            THHV_VP_REG_DS_BASE,
            THHV_VP_REG_DS_LIM,
            THHV_VP_REG_DS_AR,
            &sregs.ds,
        ));
        regs.extend_from_slice(&segment_regs(
            THHV_VP_REG_ES_SEL,
            THHV_VP_REG_ES_BASE,
            THHV_VP_REG_ES_LIM,
            THHV_VP_REG_ES_AR,
            &sregs.es,
        ));
        regs.extend_from_slice(&segment_regs(
            THHV_VP_REG_FS_SEL,
            THHV_VP_REG_FS_BASE,
            THHV_VP_REG_FS_LIM,
            THHV_VP_REG_FS_AR,
            &sregs.fs,
        ));
        regs.extend_from_slice(&segment_regs(
            THHV_VP_REG_GS_SEL,
            THHV_VP_REG_GS_BASE,
            THHV_VP_REG_GS_LIM,
            THHV_VP_REG_GS_AR,
            &sregs.gs,
        ));
        regs.extend_from_slice(&segment_regs(
            THHV_VP_REG_SS_SEL,
            THHV_VP_REG_SS_BASE,
            THHV_VP_REG_SS_LIM,
            THHV_VP_REG_SS_AR,
            &sregs.ss,
        ));
        regs.extend_from_slice(&segment_regs(
            THHV_VP_REG_TR_SEL,
            THHV_VP_REG_TR_BASE,
            THHV_VP_REG_TR_LIM,
            THHV_VP_REG_TR_AR,
            &sregs.tr,
        ));
        regs.extend_from_slice(&segment_regs(
            THHV_VP_REG_LDTR_SEL,
            THHV_VP_REG_LDTR_BASE,
            THHV_VP_REG_LDTR_LIM,
            THHV_VP_REG_LDTR_AR,
            &sregs.ldt,
        ));
        regs.extend_from_slice(&[
            reg(THHV_VP_REG_GDTR_BASE, sregs.gdt.base),
            reg(THHV_VP_REG_GDTR_LIM, sregs.gdt.limit as u64),
            reg(THHV_VP_REG_IDTR_BASE, sregs.idt.base),
            reg(THHV_VP_REG_IDTR_LIM, sregs.idt.limit as u64),
        ]);
        self.set_reg_values(&regs)
            .map_err(|e| HypervisorCpuError::SetSpecialRegs(anyhow!(e.to_string())))
    }

    fn get_fpu(&self) -> cpu::Result<FpuState> {
        Err(HypervisorCpuError::GetFloatingPointRegs(anyhow!(
            "not supported"
        )))
    }

    fn set_fpu(&self, _fpu: &FpuState) -> cpu::Result<()> {
        Ok(())
    }

    fn set_cpuid2(&self, cpuid: &[CpuIdEntry]) -> cpu::Result<()> {
        let mut guard = self.cpuid.lock().unwrap();
        *guard = cpuid.to_vec();
        Ok(())
    }

    fn enable_hyperv_synic(&self) -> cpu::Result<()> {
        Ok(())
    }

    fn get_cpuid2(&self, _num_entries: usize) -> cpu::Result<Vec<CpuIdEntry>> {
        Ok(Vec::new())
    }

    fn get_lapic(&self) -> cpu::Result<LapicState> {
        Ok(LapicState::default())
    }

    fn set_lapic(&self, _lapic: &LapicState) -> cpu::Result<()> {
        Ok(())
    }

    fn get_msrs(&self, _msrs: &mut Vec<MsrEntry>) -> cpu::Result<usize> {
        Ok(0)
    }

    fn set_msrs(&self, _msrs: &[MsrEntry]) -> cpu::Result<usize> {
        Ok(0)
    }

    fn get_mp_state(&self) -> cpu::Result<MpState> {
        Ok(MpState::Themis)
    }

    fn set_mp_state(&self, _mp_state: MpState) -> cpu::Result<()> {
        Ok(())
    }

    fn tsc_khz(&self) -> cpu::Result<Option<u32>> {
        // Try CPUID leaf 0x15: TSC/Crystal ratio (Intel SDM Vol. 3A §18.7.3).
        // EBX/EAX is the TSC-to-crystal multiplier/denominator; ECX is the
        // crystal frequency in Hz.  If ECX is 0 (common in older models) we
        // fall back to the Tiger/Ice-Lake nominal 19.2 MHz crystal.
        // SAFETY: CPUID is always available on x86_64.
        let leaf15 = unsafe { std::arch::x86_64::__cpuid(0x15) };
        if leaf15.eax != 0 && leaf15.ebx != 0 {
            let crystal_hz = if leaf15.ecx != 0 {
                leaf15.ecx as u64
            } else {
                19_200_000u64
            };
            let tsc_khz =
                (crystal_hz * leaf15.ebx as u64 / leaf15.eax as u64 / 1000) as u32;
            if tsc_khz > 0 {
                return Ok(Some(tsc_khz));
            }
        }

        // Try CPUID leaf 0x16: Processor Frequency Information.
        // EAX[15:0] = processor base frequency in MHz.
        let leaf16 = unsafe { std::arch::x86_64::__cpuid(0x16) };
        if (leaf16.eax & 0xffff) != 0 {
            return Ok(Some((leaf16.eax & 0xffff) * 1000));
        }

        Ok(None)
    }

    fn state(&self) -> cpu::Result<CpuState> {
        Err(HypervisorCpuError::GetCpuid(anyhow!(
            "state save not supported"
        )))
    }

    fn set_state(&self, _state: &CpuState) -> cpu::Result<()> {
        Err(HypervisorCpuError::SetCpuid(anyhow!(
            "state restore not supported"
        )))
    }

    fn run(&mut self) -> std::result::Result<VmExit, HypervisorCpuError> {
        self.vm_state
            .ensure_initialized()
            .map_err(|e| HypervisorCpuError::RunVcpu(e.into()))?;

        let mut run = ThhvRunVp::default();
        match ioctl_with_mut_ref(self.fd.as_raw_fd(), THHV_RUN_VP, &mut run) {
            Ok(()) => self.handle_run_exit(run),
            Err(err) => match err.raw_os_error() {
                Some(libc::EAGAIN) | Some(libc::EINTR) => Ok(VmExit::Ignore),
                _ => Err(HypervisorCpuError::RunVcpu(err.into())),
            },
        }
    }

    fn translate_gva(&self, _gva: u64, _flags: u64) -> cpu::Result<(u64, u32)> {
        Err(HypervisorCpuError::TranslateVirtualAddress(anyhow!(
            "not supported"
        )))
    }

    fn boot_msr_entries(&self) -> &'static [MsrEntry] {
        &EMPTY_BOOT_MSRS
    }

    fn nmi(&self) -> cpu::Result<()> {
        Err(HypervisorCpuError::Nmi(anyhow!("not supported")))
    }
}

impl ThemisVcpu {
    fn get_reg_values(&self, names: &[u64]) -> cpu::Result<Vec<u64>> {
        let mut regs: Vec<ThhvRegNameValue> =
            names.iter().copied().map(|name| reg(name, 0)).collect();
        let mut header = ThhvVpRegisters {
            count: regs.len() as u32,
            rsvd: 0,
            regs: regs.as_mut_ptr() as usize as u64,
        };
        ioctl_with_mut_ref(self.fd.as_raw_fd(), THHV_GET_VP_STATE, &mut header)
            .map_err(|e| HypervisorCpuError::GetRegister(e.into()))?;
        Ok(regs.into_iter().map(|entry| entry.value).collect())
    }

    fn set_reg_values(&self, regs: &[ThhvRegNameValue]) -> cpu::Result<()> {
        let mut regs = regs.to_vec();
        let mut header = ThhvVpRegisters {
            count: regs.len() as u32,
            rsvd: 0,
            regs: regs.as_mut_ptr() as usize as u64,
        };
        ioctl_with_mut_ref(self.fd.as_raw_fd(), THHV_SET_VP_STATE, &mut header)
            .map_err(|e| HypervisorCpuError::SetRegister(e.into()))
    }

    fn handle_run_exit(
        &mut self,
        run: ThhvRunVp,
    ) -> std::result::Result<VmExit, HypervisorCpuError> {
        // SAFETY: msg_buf holds a themic_intercept_message prefix written by the kernel.
        let msg =
            unsafe { ptr::read_unaligned(run.msg_buf.as_ptr() as *const ThemicInterceptMessage) };

        if msg.header.message_type == THEMIC_MSG_SHUTDOWN {
            return Ok(VmExit::Shutdown);
        }

        // Temporary: log exit reason distribution
        static EXIT_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let en = EXIT_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Always log AP exits (vp_index > 0) for the first 20 + every 5000th
        let is_ap = self._vp_index > 0;
        if en < 30 || en % 500 == 0 || (is_ap && (en < 39550 || en % 5000 == 0)) {
            eprintln!(
                "[THEMIS-EXIT] #{en} vp={} reason={} rip={:#x} msr_num={:#x}",
                self._vp_index, msg.exit_reason, msg.guest_rip, msg.msr_number
            );
        }

        // Exit reason 33 = VM-entry failure due to invalid guest state.
        // Capavisor should have already panicked with a full VMCS dump, but
        // if we somehow reach here it means the capavisor build is stale.
        // Panic here too so the bug is never silently ignored.
        const EXIT_REASON_VMENTRY_INVALID_GUEST: u32 = 33;
        if msg.exit_reason == EXIT_REASON_VMENTRY_INVALID_GUEST {
            panic!(
                "[THEMIS] exit reason 33 (invalid guest state) reached CHV — \
                 capavisor VMCS dump should be in serial log above. \
                 vp={} rip={:#x} qual={:#x}",
                self._vp_index, msg.guest_rip, msg.exit_qualification
            );
        }

        match msg.exit_reason {
            EXIT_REASON_EXCEPTION_NMI
            | EXIT_REASON_EXTERNAL_INTERRUPT
            | EXIT_REASON_VMCALL
            | EXIT_REASON_RDMSR => Ok(VmExit::Ignore),
            EXIT_REASON_WRMSR => {
                self.handle_wrmsr_exit(&msg);
                Ok(VmExit::Ignore)
            }
            EXIT_REASON_TRIPLE_FAULT => {
                let cr0  = self.get_reg_values(&[THHV_VP_REG_CR0]).unwrap_or_default();
                let cr3  = self.get_reg_values(&[THHV_VP_REG_CR3]).unwrap_or_default();
                let cr4  = self.get_reg_values(&[THHV_VP_REG_CR4]).unwrap_or_default();
                let efer = self.get_reg_values(&[THHV_VP_REG_EFER]).unwrap_or_default();
                let rax  = self.get_reg_values(&[THHV_VP_REG_RAX]).unwrap_or_default();
                eprintln!("[TRIPLE-FAULT] rip={:#x} cr0={:#x} cr3={:#x} cr4={:#x} efer={:#x} rax={:#x}",
                    msg.guest_rip,
                    cr0.first().copied().unwrap_or(0),
                    cr3.first().copied().unwrap_or(0),
                    cr4.first().copied().unwrap_or(0),
                    efer.first().copied().unwrap_or(0),
                    rax.first().copied().unwrap_or(0));
                Ok(VmExit::Reset)
            }
            EXIT_REASON_CPUID => {
                self.handle_cpuid_exit(&msg)?;
                Ok(VmExit::Ignore)
            }
            EXIT_REASON_CR_ACCESS => {
                self.handle_cr_access_exit(&msg)?;
                Ok(VmExit::Ignore)
            }
            // HLT: dom1 is idle (waiting for interrupt). The kernel driver already
            // drained the domcomm RX ring (signaling any pending ioeventfds) before
            // returning from THHV_RUN_VP. Return Ignore so the vCPU run loop retries;
            // do_switch in capavisor will inject any pending interrupt via VMENTRY_INTR_INFO.
            EXIT_REASON_HLT => Ok(VmExit::Ignore),
            EXIT_REASON_IO_INSTRUCTION => {
                self.handle_io_exit(&msg)?;
                Ok(VmExit::Ignore)
            }
            EXIT_REASON_EPT_VIOLATION => {
                self.handle_mmio_exit(&msg)?;
                Ok(VmExit::Ignore)
            }
            EXIT_REASON_APIC_ACCESS => {
                self.handle_apic_access_exit(&msg);
                Ok(VmExit::Ignore)
            }
            _ => Ok(VmExit::Ignore),
        }
    }

    /// Handle WRMSR exit — specifically IA32_TSC_DEADLINE (0x6E0) for LAPIC
    /// timer emulation.  The guest wrote a TSC deadline value; we arm a
    /// timerfd so the interrupt fires at the right wall-clock time.
    ///
    /// TSC_KHZ must match the value exposed to the guest via CPUID leaf 0x15.
    fn handle_wrmsr_exit(&self, msg: &ThemicInterceptMessage) {
        const IA32_TSC_DEADLINE: u32 = 0x6E0;
        const TSC_KHZ: u64 = 3_000_000; // 3 GHz — matches CPUID 0x15

        // Log ALL WRMSR exits (first 20 + every 100th) to verify trapping works.
        use std::sync::atomic::{AtomicU64, Ordering};
        static WRMSR_COUNT: AtomicU64 = AtomicU64::new(0);
        let n = WRMSR_COUNT.fetch_add(1, Ordering::Relaxed);
        if n < 20 || n % 100 == 0 {
            eprintln!(
                "[THEMIS-MSR] #{n} WRMSR msr={:#x} val={:#x} rip={:#x}",
                msg.msr_number, msg.msr_value, msg.guest_rip
            );
        }

        if msg.msr_number != IA32_TSC_DEADLINE {
            return;
        }

        let deadline_tsc = msg.msr_value;

        if deadline_tsc == 0 {
            // Disarm: guest cancelled the timer.
            let disarm = libc::itimerspec {
                it_interval: libc::timespec { tv_sec: 0, tv_nsec: 0 },
                it_value: libc::timespec { tv_sec: 0, tv_nsec: 0 },
            };
            unsafe {
                libc::timerfd_settime(
                    self.timer_fd.as_raw_fd(), 0, &disarm, std::ptr::null_mut(),
                );
            }
            return;
        }

        let now_tsc = unsafe { std::arch::x86_64::_rdtsc() };

        // Debug: log timing for first 20 + every 100th
        if n < 20 || n % 100 == 0 {
            let diff = if deadline_tsc > now_tsc {
                deadline_tsc - now_tsc
            } else {
                0
            };
            eprintln!(
                "[THEMIS-TIMER] #{n} deadline={:#x} now={:#x} delta_tsc={} ({} us)",
                deadline_tsc, now_tsc, diff, diff / (TSC_KHZ / 1_000_000)
            );
        }

        if now_tsc >= deadline_tsc {
            // Already expired — inject immediately via the irqfd.
            // The timer thread's EventFd is backed by the same irqfd, but
            // the fastest path is to arm the timerfd with a minimal delay.
            let fire_now = libc::itimerspec {
                it_interval: libc::timespec { tv_sec: 0, tv_nsec: 0 },
                it_value: libc::timespec { tv_sec: 0, tv_nsec: 1 },
            };
            unsafe {
                libc::timerfd_settime(
                    self.timer_fd.as_raw_fd(), 0, &fire_now, std::ptr::null_mut(),
                );
            }
            return;
        }

        let delta_tsc = deadline_tsc - now_tsc;
        // delta_ns = delta_tsc * 1_000_000 / TSC_KHZ  (avoiding overflow)
        let delta_ns = delta_tsc / (TSC_KHZ / 1_000_000);

        let tv_sec = (delta_ns / 1_000_000_000) as i64;
        let tv_nsec = (delta_ns % 1_000_000_000) as i64;

        let arm = libc::itimerspec {
            it_interval: libc::timespec { tv_sec: 0, tv_nsec: 0 },
            it_value: libc::timespec {
                tv_sec,
                tv_nsec: if tv_sec == 0 && tv_nsec == 0 { 1 } else { tv_nsec },
            },
        };
        unsafe {
            libc::timerfd_settime(
                self.timer_fd.as_raw_fd(), 0, &arm, std::ptr::null_mut(),
            );
        }
    }

    /// Handle a CR_ACCESS VM exit (exit reason 28, Intel SDM §27.1 Table C-1).
    ///
    /// The exit qualification encodes:
    ///   bits  3:0  = CR number (0=CR0, 3=CR3, 4=CR4)
    ///   bits  5:4  = access type (0=MOV to CR, 1=MOV from CR, 2=CLTS, 3=LMSW)
    ///   bits 11:8  = source/dest register (Intel encoding, see below)
    ///
    /// Intel register encoding (qual bits 11:8):
    ///   0=RAX 1=RCX 2=RDX 3=RBX 4=RSP 5=RBP 6=RSI 7=RDI 8-15=R8-R15
    fn handle_cr_access_exit(&self, msg: &ThemicInterceptMessage) -> cpu::Result<()> {
        let qual    = msg.exit_qualification;
        let cr_num  = (qual & 0xF) as u32;
        let acc     = ((qual >> 4) & 0x3) as u32;
        let reg_idx = ((qual >> 8) & 0xF) as u32;

        // Map Intel qualification register index to THHV_VP_REG_* discriminant.
        // Note: VpRegister ordering differs from Intel's.
        let thhv_gpr = |idx: u32| -> u64 {
            match idx {
                0  => THHV_VP_REG_RAX,
                1  => THHV_VP_REG_RCX,
                2  => THHV_VP_REG_RDX,
                3  => THHV_VP_REG_RBX,
                4  => THHV_VP_REG_RSP,
                5  => THHV_VP_REG_RBP,
                6  => THHV_VP_REG_RSI,
                7  => THHV_VP_REG_RDI,
                8  => THHV_VP_REG_R8,
                9  => THHV_VP_REG_R9,
                10 => THHV_VP_REG_R10,
                11 => THHV_VP_REG_R11,
                12 => THHV_VP_REG_R12,
                13 => THHV_VP_REG_R13,
                14 => THHV_VP_REG_R14,
                15 => THHV_VP_REG_R15,
                _  => THHV_VP_REG_RAX,
            }
        };

        if acc == 0 {
            // MOV to CR: get source register value, write to the appropriate CR.
            let src_reg  = thhv_gpr(reg_idx);
            let src_val  = self.get_reg_values(&[src_reg])?[0];

            match cr_num {
                0 => {
                    // When enabling paging (PG bit) while EFER.LME=1, also set EFER.LMA.
                    // Preserve host-forced FIXED0 bits (bits always set in old_cr0 due to mask).
                    let old_cr0 = self.get_reg_values(&[THHV_VP_REG_CR0])?[0];
                    // Keep bits that the host must own (in FIXED0 mask) that guest wants to clear.
                    let forced_bits = old_cr0 & !src_val;
                    let new_cr0 = src_val | forced_bits;
                    let pg_bit  = 1u64 << 31;
                    let mut updates = vec![reg(THHV_VP_REG_CR0, new_cr0)];
                    if (old_cr0 & pg_bit) == 0 && (new_cr0 & pg_bit) != 0 {
                        let efer = self.get_reg_values(&[THHV_VP_REG_EFER])?[0];
                        if efer & (1 << 8) != 0 {
                            // LME set → activate LMA now that PG is going on.
                            updates.push(reg(THHV_VP_REG_EFER, efer | (1 << 10)));
                        }
                    }
                    self.set_reg_values(&updates)?;
                }
                3 => {
                    self.set_reg_values(&[reg(THHV_VP_REG_CR3, src_val)])?;
                }
                4 => {
                    // Preserve VMXE (bit 13) — guest cannot clear it.
                    self.set_reg_values(&[reg(THHV_VP_REG_CR4, src_val | (1u64 << 13))])?;
                }
                8 => { /* CR8 / TPR — ignore */ }
                _ => {}
            }
        } else if acc == 1 {
            // MOV from CR: read CR value and write to the destination register.
            let cr_reg = match cr_num {
                0 => Some(THHV_VP_REG_CR0),
                3 => Some(THHV_VP_REG_CR3),
                4 => Some(THHV_VP_REG_CR4),
                _ => None,
            };
            if let Some(r) = cr_reg {
                let cr_val  = self.get_reg_values(&[r])?[0];
                self.set_reg_values(&[reg(thhv_gpr(reg_idx), cr_val)])?;
            }
        }
        // CLTS (acc==2) and LMSW (acc==3) are rare; continue without touching RIP.
        // RIP was already advanced by the capavisor before forwarding the exit.

        Ok(())
    }

    fn handle_cpuid_exit(&self, msg: &ThemicInterceptMessage) -> cpu::Result<()> {
        let leaf = msg.cpuid_rax as u32;
        let subleaf = msg.cpuid_rcx as u32;

        // Use the CPUID policy set by CHV (via set_cpuid2), which applies correct
        // masking for vCPU count, topology, VMX hiding, etc.
        // Fall back to raw hardware CPUID only if no policy has been set yet
        // (e.g. early firmware execution before CHV installs the policy).
        let (eax, ebx, ecx, edx) = {
            let guard = self.cpuid.lock().unwrap();
            if guard.is_empty() {
                // No policy yet — pass through host CPUID as-is.
                // SAFETY: CPUID is always available on x86_64 and has no side effects.
                let r = unsafe { std::arch::x86_64::__cpuid_count(leaf, subleaf) };
                (r.eax, r.ebx, r.ecx, r.edx)
            } else {
                // Search stored policy. Most leaves use index=0 only; indexed
                // sub-leaf leaves (0x4, 0x7, 0xB, 0x1F, 0x8000001D) store each
                // sub-leaf separately.
                //
                // For topology enumeration leaves (0xB, 0x1F) we must NOT fall
                // back to sub-leaf 0 for an unknown sub-leaf: those leaves signal
                // end-of-enumeration with ECX[15:8]=0 (level_type=0), which only
                // appears when the caller queries a sub-leaf beyond the last valid
                // one and gets all-zeros back.  Returning the sub-leaf-0 entry
                // (ECX[15:8]≠0) for every missing sub-leaf causes Linux's topology
                // scan to loop forever.
                const INDEXED_LEAVES: &[u32] = &[0x4, 0xb, 0x1f, 0x8000_001d];
                let entry = guard
                    .iter()
                    .find(|e| e.function == leaf && e.index == subleaf)
                    .or_else(|| {
                        if INDEXED_LEAVES.contains(&leaf) {
                            None // return (0,0,0,0) → end-of-enumeration
                        } else {
                            guard.iter().find(|e| e.function == leaf && e.index == 0)
                        }
                    })
                    .copied();
                match entry {
                    Some(e) => (e.eax, e.ebx, e.ecx, e.edx),
                    None => (0, 0, 0, 0),
                }
            }
        };

        self.set_reg_values(&[
            reg(THHV_VP_REG_RAX, u64::from(eax)),
            reg(THHV_VP_REG_RBX, u64::from(ebx)),
            reg(THHV_VP_REG_RCX, u64::from(ecx)),
            reg(THHV_VP_REG_RDX, u64::from(edx)),
            // RIP already advanced by capavisor before forwarding the exit.
        ])?;
        Ok(())
    }

    fn handle_io_exit(&self, msg: &ThemicInterceptMessage) -> cpu::Result<()> {
        let len = usize::from(msg.access_size);
        if len == 0 || len > 4 {
            return Ok(());
        }

        // Debug: log first few IO exits to verify the path works.
        static IO_LOG_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = IO_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n < 20 || (n < 200 && n % 50 == 0) {
            eprintln!(
                "[THEMIS-IO] #{} port=0x{:x} size={} write={} rax=0x{:x}",
                n, msg.port_number, len, msg.is_write, msg.rax
            );
        }

        if msg.is_write != 0 {
            if let Some(vm_ops) = &self.vm_ops {
                let data = msg.rax.to_le_bytes();
                vm_ops
                    .pio_write(u64::from(msg.port_number), &data[..len])
                    .map_err(|e| HypervisorCpuError::RunVcpu(e.into()))?;
            }
            // RIP already advanced by capavisor.
            return Ok(());
        }

        let mut data = [0u8; 4];
        if let Some(vm_ops) = &self.vm_ops {
            vm_ops
                .pio_read(u64::from(msg.port_number), &mut data[..len])
                .map_err(|e| HypervisorCpuError::RunVcpu(e.into()))?;
        }

        let v = u32::from_le_bytes(data);
        let mask = 0xffff_ffffu32 >> (32 - len * 8);
        let eax = (msg.rax as u32 & !mask) | (v & mask);
        // Only set RAX with the read result; RIP already advanced by capavisor.
        self.set_reg_values(&[reg(THHV_VP_REG_RAX, eax as u64)])
    }

    fn handle_mmio_exit(&mut self, msg: &ThemicInterceptMessage) -> cpu::Result<()> {
        // LAPIC MMIO fast-path: when VIRTUALIZE_APIC_ACCESSES is not
        // supported, LAPIC accesses cause EPT violations.  Handle them
        // here with per-vCPU software LAPIC state + instruction decode.
        let gpa = msg.guest_physical_address;
        if gpa >= 0xFEE0_0000 && gpa < 0xFEE0_1000 {
            return self.handle_lapic_mmio(msg);
        }

        let is_exec = msg.exit_qualification & EPT_VIOLATION_EXECUTE != 0;
        let ept_entry_present = (msg.exit_qualification >> 3) & 0x7 != 0;

        if is_exec && !ept_entry_present {
            return Err(HypervisorCpuError::RunVcpu(
                anyhow::anyhow!(
                    "EPT execute fault on unmapped GPA {:#x} (RIP {:#x} qual={:#x})",
                    msg.guest_physical_address,
                    msg.guest_rip,
                    msg.exit_qualification,
                )
                .into(),
            ));
        }

        // Use the iced-x86 emulator to decode and emulate the faulting instruction.
        // This handles MOV, MOVZX, CMP, MOVS, STOS, OR, etc. — far more than
        // the old hand-rolled decoder in the capavisor.
        let mut ctx = emulator::ThemisEmulatorContext {
            vcpu: self,
            mmio_gpa: msg.guest_physical_address,
            insn_bytes: msg.instruction_bytes,
        };

        let old_rip = msg.guest_rip;

        let mut emu = crate::arch::x86::emulator::Emulator::new(&mut ctx);
        let new_state = emu
            .emulate_first_insn(0, &msg.instruction_bytes)
            .map_err(|e| {
                eprintln!(
                    "[MMIO-EMU] emulation failed: gpa={:#x} rip={:#x} insn={:02x?}: {e}",
                    msg.guest_physical_address, msg.guest_rip, &msg.instruction_bytes[..8],
                );
                HypervisorCpuError::RunVcpu(anyhow::anyhow!("MMIO emulation failed: {e}").into())
            })?;

        // Debug: log first few MMIO emulations
        static MMIO_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let mn = MMIO_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let new_rip = match &new_state.regs {
            crate::StandardRegisters::Themis(r) => r.rip,
            #[allow(unreachable_patterns)]
            _ => 0,
        };
        if mn < 20 || (mn < 500 && mn % 50 == 0) || mn % 1000 == 0 {
            eprintln!(
                "[MMIO-DBG] #{mn} gpa={:#x} old_rip={:#x} new_rip={:#x} insn={:02x?}",
                msg.guest_physical_address, old_rip, new_rip, &msg.instruction_bytes[..6],
            );
        }

        // Write back GP registers + RIP (MMIO instructions don't modify CRs/sregs).
        self.set_regs(&new_state.regs)
            .map_err(|e| HypervisorCpuError::RunVcpu(anyhow::anyhow!("set_regs: {e}").into()))?;

        Ok(())
    }

    // ── Software LAPIC emulation (EPT-violation fallback) ──────────── //

    /// Handle an EPT violation at the LAPIC MMIO range (0xFEE00000-0xFEE00FFF).
    ///
    /// When hardware VIRTUALIZE_APIC_ACCESSES is not available, the LAPIC page
    /// is left unmapped in the child EPT.  All accesses cause EPT violations
    /// that arrive here.  We decode the faulting instruction with iced-x86,
    /// emulate the LAPIC register read/write, and advance RIP.
    fn handle_lapic_mmio(&mut self, msg: &ThemicInterceptMessage) -> cpu::Result<()> {
        use iced_x86::{Decoder, DecoderOptions, OpKind};

        let offset = (msg.guest_physical_address & 0xFFF) as u32;
        let is_write = (msg.exit_qualification >> 1) & 1 != 0;

        // Decode the faulting instruction.
        let mut decoder = Decoder::with_ip(
            64,
            &msg.instruction_bytes,
            msg.guest_rip,
            DecoderOptions::NONE,
        );
        let insn = decoder.decode();
        let insn_len = insn.len() as u64;

        if insn_len == 0 {
            eprintln!(
                "[LAPIC] failed to decode insn at RIP={:#x} bytes={:02x?}",
                msg.guest_rip,
                &msg.instruction_bytes[..8]
            );
            return Err(HypervisorCpuError::RunVcpu(
                anyhow::anyhow!("LAPIC MMIO: instruction decode failed").into(),
            ));
        }

        // Read current GP register state from the guest VMCS.
        let gregs = self
            .get_regs()
            .map_err(|e| HypervisorCpuError::RunVcpu(anyhow::anyhow!("get_regs: {e}").into()))?;
        let mut gregs = match gregs {
            crate::StandardRegisters::Themis(r) => r,
            #[allow(unreachable_patterns)]
            _ => {
                return Err(HypervisorCpuError::RunVcpu(
                    anyhow::anyhow!("unexpected register type").into(),
                ))
            }
        };

        if is_write {
            // Extract the value being written from the source operand.
            // Typical instruction: MOV dword [mem], reg  (op0=mem, op1=reg)
            //                  or: MOV dword [mem], imm  (op0=mem, op1=imm)
            let value = if insn.op_count() >= 2 {
                match insn.op1_kind() {
                    OpKind::Register => {
                        Self::read_iced_reg(&gregs, insn.op1_register()) as u32
                    }
                    OpKind::Immediate32 | OpKind::Immediate32to64 => insn.immediate32(),
                    OpKind::Immediate8 | OpKind::Immediate8to32 => insn.immediate8to32() as u32,
                    OpKind::Immediate16 => insn.immediate16() as u32,
                    _ => {
                        eprintln!(
                            "[LAPIC] unhandled write operand kind {:?} at RIP={:#x}",
                            insn.op1_kind(),
                            msg.guest_rip
                        );
                        0
                    }
                }
            } else {
                eprintln!(
                    "[LAPIC] write with {} operands at RIP={:#x}",
                    insn.op_count(),
                    msg.guest_rip
                );
                0
            };

            // Store in per-vCPU LAPIC register file.
            {
                let mut lapic = self.vm_state.lapic_regs.lock().unwrap();
                let vp = self._vp_index as usize;
                if vp < lapic.len() {
                    lapic[vp][(offset / 4) as usize] = value;
                }
            }

            // Handle side effects for specific registers.
            match offset {
                0x300 => {
                    // ICR low write — detect INIT/SIPI.
                    // Read ICR_HIGH from internal LAPIC state.
                    let icr_high = {
                        let lapic = self.vm_state.lapic_regs.lock().unwrap();
                        let vp = self._vp_index as usize;
                        if vp < lapic.len() { lapic[vp][0x310 / 4] } else { 0 }
                    };
                    self.deliver_ipi(value, icr_high);
                }
                0x0B0 => {
                    // EOI — no action needed; interrupt injection uses VMENTRY.
                }
                _ => {}
            }

            // Debug: log first few LAPIC writes.
            static LAPIC_W_LOG: std::sync::atomic::AtomicU32 =
                std::sync::atomic::AtomicU32::new(0);
            let n = LAPIC_W_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 30 || n % 500 == 0 {
                eprintln!(
                    "[LAPIC-W] #{n} vp={} offset={:#x} value={:#x}",
                    self._vp_index, offset, value
                );
            }
        } else {
            // Read: return per-vCPU register value.
            let value = {
                let lapic = self.vm_state.lapic_regs.lock().unwrap();
                let vp = self._vp_index as usize;
                if vp < lapic.len() {
                    lapic[vp][(offset / 4) as usize]
                } else {
                    0
                }
            };

            // Store into destination register (op0 for MOV reg, [mem]).
            if insn.op_count() >= 1 && insn.op0_kind() == OpKind::Register {
                Self::write_iced_reg(&mut gregs, insn.op0_register(), value as u64);
            }

            static LAPIC_R_LOG: std::sync::atomic::AtomicU32 =
                std::sync::atomic::AtomicU32::new(0);
            let n = LAPIC_R_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 30 || n % 500 == 0 {
                eprintln!(
                    "[LAPIC-R] #{n} vp={} offset={:#x} value={:#x}",
                    self._vp_index, offset, value
                );
            }
        }

        // Advance RIP past the faulting instruction.
        gregs.rip += insn_len;
        self.set_regs(&crate::StandardRegisters::Themis(gregs))
            .map_err(|e| HypervisorCpuError::RunVcpu(anyhow::anyhow!("set_regs: {e}").into()))?;

        Ok(())
    }

    /// Read a guest GPR value by iced-x86 Register enum.
    fn read_iced_reg(gregs: &ThemisStandardRegisters, r: iced_x86::Register) -> u64 {
        use iced_x86::Register::*;
        match r {
            RAX | EAX | AX | AL => gregs.rax,
            RBX | EBX | BX | BL => gregs.rbx,
            RCX | ECX | CX | CL => gregs.rcx,
            RDX | EDX | DX | DL => gregs.rdx,
            RSI | ESI | SI | SIL => gregs.rsi,
            RDI | EDI | DI | DIL => gregs.rdi,
            RSP | ESP | SP => gregs.rsp,
            RBP | EBP | BP => gregs.rbp,
            R8 | R8D | R8W | R8L => gregs.r8,
            R9 | R9D | R9W | R9L => gregs.r9,
            R10 | R10D | R10W | R10L => gregs.r10,
            R11 | R11D | R11W | R11L => gregs.r11,
            R12 | R12D | R12W | R12L => gregs.r12,
            R13 | R13D | R13W | R13L => gregs.r13,
            R14 | R14D | R14W | R14L => gregs.r14,
            R15 | R15D | R15W | R15L => gregs.r15,
            // High-byte registers: extract bits 15:8
            AH => (gregs.rax >> 8) & 0xFF,
            BH => (gregs.rbx >> 8) & 0xFF,
            CH => (gregs.rcx >> 8) & 0xFF,
            DH => (gregs.rdx >> 8) & 0xFF,
            _ => {
                eprintln!("[LAPIC] read_iced_reg: unhandled register {:?}", r);
                0
            }
        }
    }

    /// Write a value into a guest GPR by iced-x86 Register enum.
    fn write_iced_reg(gregs: &mut ThemisStandardRegisters, r: iced_x86::Register, val: u64) {
        use iced_x86::Register::*;
        match r {
            RAX => gregs.rax = val,
            EAX => gregs.rax = val & 0xFFFF_FFFF, // 32-bit writes zero-extend
            AX => gregs.rax = (gregs.rax & !0xFFFF) | (val & 0xFFFF),
            AL => gregs.rax = (gregs.rax & !0xFF) | (val & 0xFF),
            AH => gregs.rax = (gregs.rax & !0xFF00) | ((val & 0xFF) << 8),
            RBX => gregs.rbx = val,
            EBX => gregs.rbx = val & 0xFFFF_FFFF,
            BX => gregs.rbx = (gregs.rbx & !0xFFFF) | (val & 0xFFFF),
            BL => gregs.rbx = (gregs.rbx & !0xFF) | (val & 0xFF),
            BH => gregs.rbx = (gregs.rbx & !0xFF00) | ((val & 0xFF) << 8),
            RCX => gregs.rcx = val,
            ECX => gregs.rcx = val & 0xFFFF_FFFF,
            CX => gregs.rcx = (gregs.rcx & !0xFFFF) | (val & 0xFFFF),
            CL => gregs.rcx = (gregs.rcx & !0xFF) | (val & 0xFF),
            CH => gregs.rcx = (gregs.rcx & !0xFF00) | ((val & 0xFF) << 8),
            RDX => gregs.rdx = val,
            EDX => gregs.rdx = val & 0xFFFF_FFFF,
            DX => gregs.rdx = (gregs.rdx & !0xFFFF) | (val & 0xFFFF),
            DL => gregs.rdx = (gregs.rdx & !0xFF) | (val & 0xFF),
            DH => gregs.rdx = (gregs.rdx & !0xFF00) | ((val & 0xFF) << 8),
            RSI => gregs.rsi = val,
            ESI => gregs.rsi = val & 0xFFFF_FFFF,
            SI => gregs.rsi = (gregs.rsi & !0xFFFF) | (val & 0xFFFF),
            SIL => gregs.rsi = (gregs.rsi & !0xFF) | (val & 0xFF),
            RDI => gregs.rdi = val,
            EDI => gregs.rdi = val & 0xFFFF_FFFF,
            DI => gregs.rdi = (gregs.rdi & !0xFFFF) | (val & 0xFFFF),
            DIL => gregs.rdi = (gregs.rdi & !0xFF) | (val & 0xFF),
            RSP => gregs.rsp = val,
            ESP => gregs.rsp = val & 0xFFFF_FFFF,
            SP => gregs.rsp = (gregs.rsp & !0xFFFF) | (val & 0xFFFF),
            SPL => gregs.rsp = (gregs.rsp & !0xFF) | (val & 0xFF),
            RBP => gregs.rbp = val,
            EBP => gregs.rbp = val & 0xFFFF_FFFF,
            BP => gregs.rbp = (gregs.rbp & !0xFFFF) | (val & 0xFFFF),
            BPL => gregs.rbp = (gregs.rbp & !0xFF) | (val & 0xFF),
            R8 => gregs.r8 = val,
            R8D => gregs.r8 = val & 0xFFFF_FFFF,
            R8W => gregs.r8 = (gregs.r8 & !0xFFFF) | (val & 0xFFFF),
            R8L => gregs.r8 = (gregs.r8 & !0xFF) | (val & 0xFF),
            R9 => gregs.r9 = val,
            R9D => gregs.r9 = val & 0xFFFF_FFFF,
            R9W => gregs.r9 = (gregs.r9 & !0xFFFF) | (val & 0xFFFF),
            R9L => gregs.r9 = (gregs.r9 & !0xFF) | (val & 0xFF),
            R10 => gregs.r10 = val,
            R10D => gregs.r10 = val & 0xFFFF_FFFF,
            R10W => gregs.r10 = (gregs.r10 & !0xFFFF) | (val & 0xFFFF),
            R10L => gregs.r10 = (gregs.r10 & !0xFF) | (val & 0xFF),
            R11 => gregs.r11 = val,
            R11D => gregs.r11 = val & 0xFFFF_FFFF,
            R11W => gregs.r11 = (gregs.r11 & !0xFFFF) | (val & 0xFFFF),
            R11L => gregs.r11 = (gregs.r11 & !0xFF) | (val & 0xFF),
            R12 => gregs.r12 = val,
            R12D => gregs.r12 = val & 0xFFFF_FFFF,
            R12W => gregs.r12 = (gregs.r12 & !0xFFFF) | (val & 0xFFFF),
            R12L => gregs.r12 = (gregs.r12 & !0xFF) | (val & 0xFF),
            R13 => gregs.r13 = val,
            R13D => gregs.r13 = val & 0xFFFF_FFFF,
            R13W => gregs.r13 = (gregs.r13 & !0xFFFF) | (val & 0xFFFF),
            R13L => gregs.r13 = (gregs.r13 & !0xFF) | (val & 0xFF),
            R14 => gregs.r14 = val,
            R14D => gregs.r14 = val & 0xFFFF_FFFF,
            R14W => gregs.r14 = (gregs.r14 & !0xFFFF) | (val & 0xFFFF),
            R14L => gregs.r14 = (gregs.r14 & !0xFF) | (val & 0xFF),
            R15 => gregs.r15 = val,
            R15D => gregs.r15 = val & 0xFFFF_FFFF,
            R15W => gregs.r15 = (gregs.r15 & !0xFFFF) | (val & 0xFFFF),
            R15L => gregs.r15 = (gregs.r15 & !0xFF) | (val & 0xFF),
            _ => {
                eprintln!("[LAPIC] write_iced_reg: unhandled register {:?}", r);
            }
        }
    }

    /// Deliver an IPI based on the ICR low value.
    /// Handles IPI delivery modes: Fixed (0), INIT (5), and SIPI (6).
    fn deliver_ipi(&self, icr_low: u32, icr_high: u32) {
        let delivery_mode = (icr_low >> 8) & 0x7;
        let vector = icr_low & 0xFF;
        let dest_shorthand = (icr_low >> 18) & 0x3;

        // Destination APIC ID from ICR_HIGH bits [31:24], passed by capavisor
        // from VAPIC[0x310].
        let dest_apic_id = (icr_high >> 24) & 0xFF;

        eprintln!(
            "[LAPIC-IPI] vp={} icr_low={:#x} mode={} vector={:#x} shorthand={} dest_apic={}",
            self._vp_index, icr_low, delivery_mode, vector, dest_shorthand, dest_apic_id
        );

        match delivery_mode {
            0 | 1 => {
                // Fixed (0) or Lowest Priority (1) — inject vector into target VP.
                self.deliver_fixed_ipi(vector as u8, dest_shorthand, dest_apic_id);
            }
            5 => {
                // INIT — AP is already in wait-for-SIPI (set at create_vcpu).
                eprintln!("[LAPIC-IPI] INIT IPI — no-op (AP in wait-for-SIPI)");
            }
            6 => {
                // Startup IPI (SIPI) — wake target AP with CS:IP from vector.
                self.deliver_sipi(vector);
            }
            _ => {
                eprintln!(
                    "[LAPIC-IPI] unhandled delivery mode {} — ignored",
                    delivery_mode
                );
            }
        }
    }

    /// Deliver a Fixed-mode IPI: inject vector into target VP(s) via THHV_INJECT_INTERRUPT.
    fn deliver_fixed_ipi(&self, vector: u8, dest_shorthand: u32, dest_apic_id: u32) {
        let fds = self.vm_state.vp_fds.lock().unwrap();
        let num_vps = fds.len();

        // Determine target VP indices based on destination shorthand.
        let targets: Vec<usize> = match dest_shorthand {
            0 => {
                // No shorthand — use dest_apic_id.
                // APIC ID = vp_index (simple 1:1 mapping).
                let target = dest_apic_id as usize;
                if target < num_vps { vec![target] } else { vec![] }
            }
            1 => vec![self._vp_index as usize], // Self
            2 => (0..num_vps).collect(),         // All including self
            3 => {
                // All excluding self
                (0..num_vps).filter(|&i| i != self._vp_index as usize).collect()
            }
            _ => vec![],
        };

        let part_fd = self.vm_state.fd.as_raw_fd();
        for target_vp in targets {
            let mut ii = ThhvInjectInterrupt {
                vp_index: target_vp as u32,
                vector,
                pad: [0; 3],
            };
            let ret = unsafe {
                libc::ioctl(part_fd, THHV_INJECT_INTERRUPT as libc::c_ulong, &mut ii)
            };
            if ret < 0 {
                let err = std::io::Error::last_os_error();
                eprintln!(
                    "[LAPIC-IPI] Fixed IPI inject failed: vp={} vector={:#x} err={:?}",
                    target_vp, vector, err
                );
            } else {
                eprintln!(
                    "[LAPIC-IPI] Fixed IPI injected: vp={} vector={:#x}",
                    target_vp, vector
                );
            }
        }
    }

    /// Deliver SIPI to all non-self vCPUs: set full real-mode state and wake AP.
    ///
    /// After SIPI, the AP must be in real mode with CS:IP from the SIPI vector
    /// and all other segments in their reset defaults.  We also clear GP regs
    /// and set RFLAGS=0x2, CR0=0 (real mode; capavisor adjusts NE/ET), CR3/CR4/EFER=0.
    fn deliver_sipi(&self, vector: u32) {
        let fds = self.vm_state.vp_fds.lock().unwrap();
        let my_id = self._vp_index as usize;

        for (target_id, &target_fd) in fds.iter().enumerate() {
            if target_id == my_id || target_fd < 0 {
                continue;
            }

            let startup_addr = (vector as u64) << 12;
            let cs_selector = (vector as u64) << 8;

            // Real-mode segment access rights: present, R/W data for data segs,
            // execute/read for CS.
            let data_ar: u64 = 0x0093; // P=1, S=1, type=3 (R/W accessed)
            let code_ar: u64 = 0x009B; // P=1, S=1, type=B (exec/read accessed)

            let regs = [
                // CS from SIPI vector
                reg(THHV_VP_REG_CS_SEL, cs_selector),
                reg(THHV_VP_REG_CS_BASE, startup_addr),
                reg(THHV_VP_REG_CS_LIM, 0xFFFF),
                reg(THHV_VP_REG_CS_AR, code_ar),
                // DS/ES/FS/GS/SS = 0:0 with 64K limit (real-mode reset)
                reg(THHV_VP_REG_DS_SEL, 0),
                reg(THHV_VP_REG_DS_BASE, 0),
                reg(THHV_VP_REG_DS_LIM, 0xFFFF),
                reg(THHV_VP_REG_DS_AR, data_ar),
                reg(THHV_VP_REG_ES_SEL, 0),
                reg(THHV_VP_REG_ES_BASE, 0),
                reg(THHV_VP_REG_ES_LIM, 0xFFFF),
                reg(THHV_VP_REG_ES_AR, data_ar),
                reg(THHV_VP_REG_FS_SEL, 0),
                reg(THHV_VP_REG_FS_BASE, 0),
                reg(THHV_VP_REG_FS_LIM, 0xFFFF),
                reg(THHV_VP_REG_FS_AR, data_ar),
                reg(THHV_VP_REG_GS_SEL, 0),
                reg(THHV_VP_REG_GS_BASE, 0),
                reg(THHV_VP_REG_GS_LIM, 0xFFFF),
                reg(THHV_VP_REG_GS_AR, data_ar),
                reg(THHV_VP_REG_SS_SEL, 0),
                reg(THHV_VP_REG_SS_BASE, 0),
                reg(THHV_VP_REG_SS_LIM, 0xFFFF),
                reg(THHV_VP_REG_SS_AR, data_ar),
                // GDT/IDT with zero base (Linux trampoline sets its own)
                reg(THHV_VP_REG_GDTR_BASE, 0),
                reg(THHV_VP_REG_GDTR_LIM, 0xFFFF),
                reg(THHV_VP_REG_IDTR_BASE, 0),
                reg(THHV_VP_REG_IDTR_LIM, 0xFFFF),
                // RIP = 0 (offset within CS segment)
                reg(THHV_VP_REG_RIP, 0),
                // RFLAGS = 0x2 (reserved bit 1 always set)
                reg(THHV_VP_REG_RFLAGS, 0x2),
                // CR0: PE=0 for real mode. The capavisor's vmcs_adjust_cr0()
                // will add NE/ET bits required by IA32_VMX_CR0_FIXED0.
                // UNRESTRICTED_GUEST exempts PE and PG from FIXED0 requirements.
                reg(THHV_VP_REG_CR0, 0x00),
                reg(THHV_VP_REG_CR3, 0),
                reg(THHV_VP_REG_CR4, 0),
                reg(THHV_VP_REG_EFER, 0),
                // Clear GP regs
                reg(THHV_VP_REG_RAX, 0),
                reg(THHV_VP_REG_RBX, 0),
                reg(THHV_VP_REG_RCX, 0),
                reg(THHV_VP_REG_RDX, 0),
                reg(THHV_VP_REG_RSI, 0),
                reg(THHV_VP_REG_RDI, 0),
                reg(THHV_VP_REG_RBP, 0),
                reg(THHV_VP_REG_RSP, 0),
                // Wake the AP
                reg(THHV_VP_REG_ACTIVITY_STATE, 0),
            ];
            let mut regs = regs.to_vec();
            let mut header = ThhvVpRegisters {
                count: regs.len() as u32,
                rsvd: 0,
                regs: regs.as_mut_ptr() as usize as u64,
            };
            let res = ioctl_with_mut_ref(target_fd, THHV_SET_VP_STATE, &mut header);
            eprintln!(
                "[LAPIC-IPI] SIPI vector={:#x} → vCPU {} CS:IP={:#x}:{:#x} result={:?}",
                vector, target_id, cs_selector, 0, res
            );
        }
    }

    /// Handle APIC-access exit forwarded by capavisor (Mode A — APICv).
    ///
    /// Only ICR low writes (offset 0x300) are forwarded.  We parse the ICR
    /// value to detect INIT (delivery mode 5) and Startup IPI (delivery mode 6).
    /// For SIPI, we configure the target AP's CS:IP from the SIPI vector and
    /// transition it from wait-for-SIPI to runnable via THHV_SET_VP_STATE.
    fn handle_apic_access_exit(&self, msg: &ThemicInterceptMessage) {
        // RAX = ICR_LOW value (decoded by capavisor from guest instruction).
        // RCX (msr_number) = ICR_HIGH value (read from VAPIC[0x310] by capavisor).
        let icr_val = msg.rax as u32;
        let icr_high = msg.msr_number;
        self.deliver_ipi(icr_val, icr_high);
    }
}

fn reg(name: u64, value: u64) -> ThhvRegNameValue {
    ThhvRegNameValue { name, value }
}

fn segment_regs(
    sel: u64,
    base: u64,
    lim: u64,
    ar: u64,
    seg: &SegmentRegister,
) -> [ThhvRegNameValue; 4] {
    [
        reg(sel, seg.selector.into()),
        reg(base, seg.base),
        reg(lim, seg.limit.into()),
        reg(ar, segment_access_rights(seg).into()),
    ]
}

fn segment_from_raw(selector: u64, base: u64, limit: u64, access_rights: u64) -> SegmentRegister {
    let ar = access_rights as u32;
    SegmentRegister {
        base,
        limit: limit as u32,
        selector: selector as u16,
        type_: (ar & 0x0f) as u8,
        s: ((ar >> 4) & 0x1) as u8,
        dpl: ((ar >> 5) & 0x3) as u8,
        present: ((ar >> 7) & 0x1) as u8,
        avl: ((ar >> 12) & 0x1) as u8,
        l: ((ar >> 13) & 0x1) as u8,
        db: ((ar >> 14) & 0x1) as u8,
        g: ((ar >> 15) & 0x1) as u8,
        unusable: ((ar >> 16) & 0x1) as u8,
    }
}

fn segment_access_rights(seg: &SegmentRegister) -> u32 {
    u32::from(seg.type_ & 0x0f)
        | (u32::from(seg.s & 0x1) << 4)
        | (u32::from(seg.dpl & 0x3) << 5)
        | (u32::from(seg.present & 0x1) << 7)
        | (u32::from(seg.avl & 0x1) << 12)
        | (u32::from(seg.l & 0x1) << 13)
        | (u32::from(seg.db & 0x1) << 14)
        | (u32::from(seg.g & 0x1) << 15)
        | (u32::from(seg.unusable & 0x1) << 16)
}

fn page_size() -> anyhow::Result<usize> {
    // SAFETY: sysconf is thread-safe for this name.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(page_size as usize)
}

fn query_meta_pages(fd: i32, query_type: u32) -> anyhow::Result<u64> {
    let mut query = ThhvQuery {
        query_type,
        rsvd: 0,
        result: 0,
    };
    ioctl_with_mut_ref(fd, THHV_QUERY, &mut query)?;
    Ok(query.result)
}

fn ioctl_noarg(fd: i32, request: c_ulong) -> std::io::Result<()> {
    // SAFETY: ioctl is called with a valid fd and no third argument.
    let ret = unsafe { libc::ioctl(fd, request) };
    if ret < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn ioctl_with_mut_ref<T>(fd: i32, request: c_ulong, arg: &mut T) -> std::io::Result<()> {
    // SAFETY: ioctl is called with a valid fd and a pointer to a live repr(C) object.
    let ret = unsafe { libc::ioctl(fd, request, arg as *mut T) };
    if ret < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn ioctl_with_mut_ref_ret_fd<T>(
    fd: i32,
    request: c_ulong,
    arg: &mut T,
) -> std::io::Result<OwnedFd> {
    // SAFETY: ioctl is called with a valid fd and a pointer to a live repr(C) object.
    let ret = unsafe { libc::ioctl(fd, request, arg as *mut T) };
    if ret < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        // SAFETY: ioctl returned a new owned fd.
        Ok(unsafe { OwnedFd::from_raw_fd(ret) })
    }
}
