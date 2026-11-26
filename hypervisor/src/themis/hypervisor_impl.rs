//! `ThemisHypervisor` — the entry point handed back from `crate::new()`
//! when CHV is built with the `themis` feature.  Owns the `/dev/thhv`
//! file descriptor and answers `is_available` / `create_vm`.

use std::fs::OpenOptions;
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::sync::{Arc, Mutex};

use super::ThemisVm;
use super::abi::{
    THHV_CREATE_PARTITION, THHV_SEND_SHARED_META, ThhvCreatePartition, ThhvInitializePartition,
};
use super::consts::{
    THEMIS_MAX_VCPUS, THHV_META_PAGES_PER_VP, THHV_META_PAGES_SHARED, THHV_QUERY_META_PAGES_PER_VP,
    THHV_QUERY_META_PAGES_SHARED, THHV_SCHED_SYNC,
};
use super::helpers::{ioctl_with_mut_ref, ioctl_with_mut_ref_ret_fd, page_size, query_meta_pages};
use super::mmap::MmapRegion;
use super::vm_state::ThemisVmState;
use crate::arch::x86::{CPUID_FLAG_VALID_INDEX, CpuIdEntry};
use crate::hypervisor::{self, Hypervisor};
use crate::{HypervisorType, HypervisorVmConfig, vm};

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

        let page_size = page_size().map_err(hypervisor::HypervisorError::HypervisorCreate)?;
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
        let themis_config = match _config.themis {
            Some(config) => config,
            None => {
                let profile = if confidential {
                    super::config::DefaultProfile::Confidential
                } else {
                    super::config::DefaultProfile::Standard
                };
                Arc::new(
                    super::config::ThemisConfig::builtin_default(profile)
                        .map_err(|error| hypervisor::HypervisorError::VmCreate(error.into()))?,
                )
            }
        };
        let mut themis_config = Arc::unwrap_or_clone(themis_config);
        themis_config.general.confidential |= confidential;
        themis_config
            .validate()
            .map_err(|error| hypervisor::HypervisorError::VmCreate(error.into()))?;
        let confidential = themis_config.general.confidential;
        let themis_config = Arc::new(themis_config);

        let mut create = ThhvCreatePartition {
            cores_mask: themis_config
                .policies
                .cores
                .resolved_mask()
                .map_err(|error| hypervisor::HypervisorError::VmCreate(error.into()))?,
            api_flags: !0,
            sched_policy: THHV_SCHED_SYNC,
            num_vps: self.get_max_vcpus(),
        };

        let part_fd =
            ioctl_with_mut_ref_ret_fd(self.fd.as_raw_fd(), THHV_CREATE_PARTITION, &mut create)
                .map_err(|e| hypervisor::HypervisorError::VmCreate(e.into()))?;

        let shared_meta = MmapRegion::new_shared_anonymous(self.shared_meta_pages * self.page_size)
            .map_err(hypervisor::HypervisorError::VmCreate)?;
        let mut init = ThhvInitializePartition {
            meta_uaddr: shared_meta.as_u64(),
            meta_size: (self.shared_meta_pages * self.page_size) as u64,
            apic_access_uaddr: 0,
            apic_access_size: 0,
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
                initialization_failed: Mutex::new(false),
                pv: std::sync::OnceLock::new(),
                device_memory: vm_memory::GuestMemoryAtomic::new(vm_memory::GuestMemoryMmap::new()),
                msr_policy_pushed: Mutex::new(false),
                page_size: self.page_size,
                vp_meta_pages: self.vp_meta_pages,
                _shared_meta: shared_meta,
                _timer_eventfd: Mutex::new(None),
                pending_memory: Mutex::new(Vec::new()),
                gsi_vectors: Mutex::new(std::collections::HashMap::new()),
                vp_fds: Mutex::new(Vec::new()),
                lapic_regs: Mutex::new(Vec::new()),
                cpuid_entries: Mutex::new(Vec::new()),
                ivshmem_bars: Mutex::new(Vec::new()),
                ioevents: Mutex::new(super::ioevent::IoEvents::default()),
                vtom_bit: if confidential {
                    // Resolution priority: themis config's `general.vtom_bit`
                    // override wins; else auto-derive from CPUID leaf
                    // 0x80000008 (MAXPHYADDR - 1).
                    themis_config.general.vtom_bit.map_or_else(
                        || {
                            // CPUID is unprivileged; capavisor intercepts
                            // hypervisor leaves but 0x80000008 is passed through.
                            (core::arch::x86_64::__cpuid(0x80000008).eax & 0xFF) - 1
                        },
                        |b| b as u32,
                    )
                } else {
                    0
                },
                confidential,
                themis_config,
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
                collect_cpuid_leaf(entries, function, std::arch::x86_64::__cpuid_count);
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

fn collect_cpuid_leaf(
    entries: &mut Vec<CpuIdEntry>,
    function: u32,
    mut read: impl FnMut(u32, u32) -> std::arch::x86_64::CpuidResult,
) {
    // Intel SDM CPUID leaf definitions; AMD extended cache/RDT leaves also
    // consume ECX. Do not infer indexedness from sampled register values.
    let indexed = matches!(
        function,
        0x4 | 0x7
            | 0xb
            | 0xd
            | 0xf
            | 0x10
            | 0x12
            | 0x14
            | 0x17
            | 0x18
            | 0x1b
            | 0x1d
            | 0x1f
            | 0x20
            | 0x23
            | 0x24
            | 0x8000_001d
            | 0x8000_0020
    );
    // Preserve the backend's existing subleaf coverage; XSAVE components
    // are sparse in a 64-bit bitmap and must not stop at an all-zero hole.
    let last_index = if function == 0xd {
        63
    } else if indexed {
        3
    } else {
        0
    };
    for index in 0..=last_index {
        let result = read(function, index);
        entries.push(CpuIdEntry {
            function,
            index,
            flags: if indexed { CPUID_FLAG_VALID_INDEX } else { 0 },
            eax: result.eax,
            ebx: result.ebx,
            ecx: result.ecx,
            edx: result.edx,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidential_runtime_rejects_legacy_transport_before_partition_creation() {
        let hypervisor = ThemisHypervisor {
            fd: std::fs::File::open("/dev/null").unwrap().into(),
            page_size: 4096,
            vp_meta_pages: 0,
            shared_meta_pages: 0,
        };
        let config = super::super::config::ThemisConfig::from_json_str(
            r#"{"general":{"device_transport":"legacy"}}"#,
        )
        .unwrap();
        let error = hypervisor
            .create_vm(HypervisorVmConfig {
                confidential: true,
                themis: Some(Arc::new(config)),
                ..Default::default()
            })
            .err()
            .expect("legacy confidential launch must fail");
        let hypervisor::HypervisorError::VmCreate(error) = error else {
            panic!("unexpected failure: {error}");
        };
        assert!(error.to_string().contains("require PV device transport"));
    }

    #[test]
    fn cpuid_index_flags_follow_leaf_semantics_not_sample_values() {
        let mut entries = Vec::new();
        for function in [0, 1, 7, 0xd, 0x8000_001d] {
            collect_cpuid_leaf(&mut entries, function, |_, index| {
                std::arch::x86_64::CpuidResult {
                    eax: if index == 3 || index == 9 { 42 } else { 0 },
                    ebx: 0,
                    ecx: 0,
                    edx: 0,
                }
            });
        }
        for function in [0, 1] {
            let leaf: Vec<_> = entries.iter().filter(|e| e.function == function).collect();
            assert_eq!(leaf.len(), 1);
            assert_eq!((leaf[0].index, leaf[0].flags), (0, 0));
        }
        for function in [7, 0xd, 0x8000_001d] {
            assert!(
                entries
                    .iter()
                    .filter(|e| e.function == function)
                    .all(|e| e.flags == CPUID_FLAG_VALID_INDEX)
            );
            assert_eq!(
                entries
                    .iter()
                    .find(|e| e.function == function && e.index == 3)
                    .unwrap()
                    .eax,
                42
            );
        }
        assert_eq!(
            entries
                .iter()
                .find(|e| e.function == 0xd && e.index == 9)
                .unwrap()
                .eax,
            42
        );
    }
}
