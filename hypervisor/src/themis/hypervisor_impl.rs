//! `ThemisHypervisor` — the entry point handed back from `crate::new()`
//! when CHV is built with the `themis` feature.  Owns the `/dev/thhv`
//! file descriptor and answers `is_available` / `create_vm`.

use std::fs::OpenOptions;
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::sync::{Arc, Mutex};

use super::abi::{
    THHV_CREATE_PARTITION, THHV_SEND_SHARED_META, ThhvCreatePartition, ThhvInitializePartition,
};
use super::consts::{
    THEMIS_MAX_VCPUS, THHV_META_PAGES_PER_VP, THHV_META_PAGES_SHARED, THHV_QUERY_META_PAGES_PER_VP,
    THHV_QUERY_META_PAGES_SHARED, THHV_SCHED_SYNC,
};
use super::helpers::{
    ioctl_with_mut_ref, ioctl_with_mut_ref_ret_fd, page_size, query_meta_pages,
};
use super::mmap::MmapRegion;
use super::vm_state::ThemisVmState;
use super::ThemisVm;
use crate::arch::x86::CpuIdEntry;
use crate::hypervisor::{self, Hypervisor};
use crate::vm;
use crate::{HypervisorType, HypervisorVmConfig};

pub struct ThemisHypervisor {
    fd: OwnedFd,
    page_size: usize,
    vp_meta_pages: usize,
    shared_meta_pages: usize,
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
        let confidential = _config.confidential;
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
                msr_policy_pushed: Mutex::new(false),
                page_size: self.page_size,
                vp_meta_pages: self.vp_meta_pages,
                _shared_meta: shared_meta,
                _apic_access: apic_access,
                _timer_eventfd: Mutex::new(None),
                pending_memory: Mutex::new(Vec::new()),
                gsi_vectors: Mutex::new(std::collections::HashMap::new()),
                vp_fds: Mutex::new(Vec::new()),
                lapic_regs: Mutex::new(Vec::new()),
                cpuid_entries: Mutex::new(Vec::new()),
                ivshmem_bars: Mutex::new(Vec::new()),
                pending_ioeventfds: Mutex::new(Vec::new()),
                vtom_bit: if confidential {
                    (core::arch::x86_64::__cpuid(0x80000008).eax & 0xFF) - 1
                } else {
                    0
                },
                confidential,
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
                    let result = std::arch::x86_64::__cpuid_count(function, index);
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

        let max_leaf = std::arch::x86_64::__cpuid(0).eax;
        collect(&mut entries, 0, max_leaf.min(0x20));

        let max_ext = std::arch::x86_64::__cpuid(0x8000_0000).eax;
        if max_ext >= 0x8000_0000 {
            collect(&mut entries, 0x8000_0000, max_ext.min(0x8000_0020));
        }

        Ok(entries)
    }

    fn get_max_vcpus(&self) -> u32 {
        THEMIS_MAX_VCPUS
    }

    fn get_guest_debug_hw_bps(&self) -> usize {
        0
    }
}
