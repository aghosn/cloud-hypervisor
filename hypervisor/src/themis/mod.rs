use std::any::Any;
use std::fs::OpenOptions;
use std::mem::size_of;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
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
const EXIT_REASON_CPUID: u32 = 10;
const EXIT_REASON_HLT: u32 = 12;
const EXIT_REASON_VMCALL: u32 = 18;
const EXIT_REASON_CR_ACCESS: u32 = 28;     // Intel SDM: MOV to/from CR0/CR3/CR4/CR8, CLTS, LMSW
const EXIT_REASON_IO_INSTRUCTION: u32 = 30; // Intel SDM: IN, INS, OUT, OUTS
const EXIT_REASON_RDMSR: u32 = 31;
const EXIT_REASON_WRMSR: u32 = 32;
const EXIT_REASON_EPT_VIOLATION: u32 = 48;

const THEMIC_MSG_SHUTDOWN: u32 = 0x0004;

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
    /// Guest memory regions recorded during create_user_memory_region but not
    /// yet sent to the domain.  THHV_SET_GUEST_MEMORY is deferred until
    /// ensure_initialized() so that cloud-hypervisor can finish writing
    /// firmware/ACPI/kernel data into the mmap'd pages while dom0 still holds
    /// EPT access.  Once SET_GUEST_MEMORY fires the capavisor removes those
    /// pages from dom0's EPT, so all writes must complete before then.
    pending_memory: Mutex<Vec<ThhvSetGuestMemory>>,
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
    exit_log_count: std::collections::HashMap<u32, u32>,
    /// CPUID policy set by CHV via set_cpuid2() before the first run.
    /// Searched by (function, index) on every CPUID exit.
    cpuid: std::sync::Mutex<Vec<CpuIdEntry>>,
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
        let mut init = ThhvInitializePartition {
            meta_uaddr: shared_meta.as_u64(),
            meta_size: (self.shared_meta_pages * self.page_size) as u64,
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
                pending_memory: Mutex::new(Vec::new()),
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
        eprintln!("[THEMIS-DBG] register_irqfd gsi={gsi} fd={}", fd.as_raw_fd());
        let mut irqfd = ThhvIrqfd {
            fd: fd.as_raw_fd(),
            gsi,
            flags: 0,
            rsvd: 0,
        };
        let r = ioctl_with_mut_ref(self.state.fd.as_raw_fd(), THHV_IRQFD, &mut irqfd)
            .map_err(|e| vm::HypervisorVmError::RegisterIrqFd(e.into()));
        eprintln!("[THEMIS-DBG] register_irqfd gsi={gsi} result={r:?}");
        r
    }

    fn unregister_irqfd(&self, fd: &EventFd, gsi: u32) -> vm::Result<()> {
        let mut irqfd = ThhvIrqfd {
            fd: fd.as_raw_fd(),
            gsi,
            flags: THHV_IRQFD_FLAG_DEASSIGN,
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

        Ok(Box::new(ThemisVcpu {
            fd: vcpu_fd,
            _vp_index: id,
            vm_state: self.state.clone(),
            vm_ops,
            _meta: meta,
            _comm: comm,
            exit_log_count: std::collections::HashMap::new(),
            cpuid: std::sync::Mutex::new(Vec::new()),
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

    fn set_gsi_routing(&self, _entries: &[IrqRoutingEntry]) -> vm::Result<()> {
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

        eprintln!("[THEMIS-DBG] set_regs: rip=0x{:x} rbx=0x{:x} rflags=0x{:x}",
            regs.rip, regs.rbx, regs.rflags);
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

        // Log first 3 occurrences of each exit reason (diagnostic; remove when stable).
        let count = self.exit_log_count.entry(msg.exit_reason).or_insert(0);
        if *count < 3 {
            eprintln!(
                "[THEMIS-DBG] exit reason={} rip={:#x} qual={:#x} port={} is_write={}",
                msg.exit_reason, msg.guest_rip, msg.exit_qualification,
                msg.port_number, msg.is_write
            );
            *count += 1;
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
            | EXIT_REASON_RDMSR
            | EXIT_REASON_WRMSR => Ok(VmExit::Ignore),
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
            _ => Ok(VmExit::Ignore),
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
                    let old_cr0 = self.get_reg_values(&[THHV_VP_REG_CR0])?[0];
                    let pg_bit  = 1u64 << 31;
                    let mut updates = vec![reg(THHV_VP_REG_CR0, src_val)];
                    if (old_cr0 & pg_bit) == 0 && (src_val & pg_bit) != 0 {
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
        // CLTS (acc==2) and LMSW (acc==3) are rare; advance RIP and continue.

        self.advance_rip(msg.guest_rip, msg.instruction_length)?;
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
                // Search stored policy. Most leaves use index=0; indexed leaves
                // (0x4, 0x7, 0xB, 0x1F, 0x8000001D) are stored with their exact
                // sub-leaf index. Fall back to all-zeros for unknown leaves
                // (matches hardware behaviour for leaves beyond max).
                let entry = guard
                    .iter()
                    .find(|e| e.function == leaf && e.index == subleaf)
                    .or_else(|| guard.iter().find(|e| e.function == leaf && e.index == 0))
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
            reg(
                THHV_VP_REG_RIP,
                msg.guest_rip.wrapping_add(u64::from(msg.instruction_length)),
            ),
        ])
    }

    fn handle_io_exit(&self, msg: &ThemicInterceptMessage) -> cpu::Result<()> {
        let len = usize::from(msg.access_size);
        if len == 0 || len > 4 {
            return Ok(());
        }

        if msg.is_write != 0 {
            if let Some(vm_ops) = &self.vm_ops {
                let data = msg.rax.to_le_bytes();
                vm_ops
                    .pio_write(u64::from(msg.port_number), &data[..len])
                    .map_err(|e| HypervisorCpuError::RunVcpu(e.into()))?;
            }
            self.advance_rip(msg.guest_rip, msg.instruction_length)?;
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
        self.advance_rip_and_set_rax(msg.guest_rip, msg.instruction_length, eax as u64)
    }

    fn handle_mmio_exit(&self, msg: &ThemicInterceptMessage) -> cpu::Result<()> {
        let is_write = msg.exit_qualification & EPT_VIOLATION_DATA_WRITE != 0;
        let is_read = msg.exit_qualification & EPT_VIOLATION_DATA_READ != 0;
        let is_exec = msg.exit_qualification & EPT_VIOLATION_EXECUTE != 0;
        // Bits 3:5 = 000 means the EPT entry has no permissions (GPA not mapped at all).
        let ept_entry_present = (msg.exit_qualification >> 3) & 0x7 != 0;

        if is_exec && !ept_entry_present {
            // Execute fault on an unmapped page — EPT setup for this region failed.
            return Err(HypervisorCpuError::RunVcpu(
                anyhow::anyhow!(
                    "EPT execute fault on unmapped GPA {:#x} (RIP {:#x} qual={:#x}): memory region not set up in child EPT",
                    msg.guest_physical_address,
                    msg.guest_rip,
                    msg.exit_qualification,
                )
                .into(),
            ));
        }

        if is_write {
            if let Some(vm_ops) = &self.vm_ops {
                let data = msg.rax.to_le_bytes();
                vm_ops
                    .mmio_write(msg.guest_physical_address, &data)
                    .map_err(|e| HypervisorCpuError::RunVcpu(e.into()))?;
            }
            self.advance_rip(msg.guest_rip, msg.instruction_length)?;
        } else if is_read {
            if let Some(vm_ops) = &self.vm_ops {
                let mut data = [0u8; 8];
                vm_ops
                    .mmio_read(msg.guest_physical_address, &mut data)
                    .map_err(|e| HypervisorCpuError::RunVcpu(e.into()))?;
                self.advance_rip_and_set_rax(
                    msg.guest_rip,
                    msg.instruction_length,
                    u64::from_le_bytes(data),
                )?;
            } else {
                self.advance_rip(msg.guest_rip, msg.instruction_length)?;
            }
        }

        Ok(())
    }

    fn advance_rip(&self, guest_rip: u64, instruction_length: u32) -> cpu::Result<()> {
        self.set_reg_values(&[reg(
            THHV_VP_REG_RIP,
            guest_rip.wrapping_add(u64::from(instruction_length)),
        )])
    }

    fn advance_rip_and_set_rax(
        &self,
        guest_rip: u64,
        instruction_length: u32,
        rax: u64,
    ) -> cpu::Result<()> {
        self.set_reg_values(&[
            reg(
                THHV_VP_REG_RIP,
                guest_rip.wrapping_add(u64::from(instruction_length)),
            ),
            reg(THHV_VP_REG_RAX, rax),
        ])
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
