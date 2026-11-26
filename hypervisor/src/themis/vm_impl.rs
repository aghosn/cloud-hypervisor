//! `ThemisVm` and its `vm::Vm` trait implementation.
//!
//! Owns the per-VM `Arc<ThemisVmState>` and translates CHV's `vm::Vm`
//! operations into THHV ioctls (memory regions, irqfd, ioeventfd, vCPU
//! creation, GSI routing).  Stub methods that are no-ops in the Themis
//! trust model carry `// Owned by capavisor:` comments.

use std::any::Any;
use std::os::fd::AsRawFd;
use std::sync::Arc;

use anyhow::anyhow;
use vfio_ioctls::VfioDeviceFd;
use vmm_sys_util::eventfd::EventFd;

use super::ThemisVcpu;
use super::abi::*;
use super::consts::*;
use super::helpers::{ioctl_with_mut_ref, ioctl_with_mut_ref_ret_fd, reg};
use super::mmap::MmapRegion;
use super::vm_state::ThemisVmState;
use crate::cpu::Vcpu;
use crate::vm::{self, DataMatch, InterruptSourceConfig, VmOps};
use crate::{IoEventAddress, IrqRoutingEntry};

pub struct ThemisVm {
    pub(super) state: Arc<ThemisVmState>,
}

impl vm::Vm for ThemisVm {
    fn shared_device_memory(&self) -> Option<vm::SharedDeviceMemory> {
        self.state
            .themis_config
            .general
            .uses_pv_devices()
            .then(|| self.state.device_memory.clone())
    }

    fn uses_guest_mp_wakeup(&self) -> bool {
        self.state.themis_config.general.uses_pv_devices()
    }

    fn set_identity_map_address(&self, _address: u64) -> vm::Result<()> {
        // KVM-specific (TSS for unrestricted-guest emulation on early VT-x).
        // Capavisor uses unrestricted guest unconditionally.  No-op.
        Ok(())
    }

    fn set_tss_address(&self, _offset: usize) -> vm::Result<()> {
        // KVM-specific (see set_identity_map_address).  No-op.
        Ok(())
    }

    fn create_irq_chip(&self) -> vm::Result<()> {
        // Owned by capavisor: the LAPIC/IOAPIC virtualization runs in
        // capavisor; there is no per-VM "irqchip" object on the CHV side.
        Ok(())
    }

    fn register_irqfd(&self, fd: &EventFd, gsi: u32) -> vm::Result<()> {
        let (vector, dest_apic) = self
            .state
            .gsi_vectors
            .lock()
            .unwrap()
            .get(&gsi)
            .copied()
            .unwrap_or((0, 0));
        let vector = vector as u32;
        log::debug!(
            "Themis register_irqfd gsi={gsi} vec={vector} vp={dest_apic} fd={}",
            fd.as_raw_fd()
        );
        let mut irqfd = ThhvIrqfd {
            fd: fd.as_raw_fd(),
            gsi,
            flags: 0,
            vector,
            vp_index: dest_apic as u32,
            rsvd: 0,
        };
        let r = ioctl_with_mut_ref(self.state.fd.as_raw_fd(), THHV_IRQFD, &mut irqfd)
            .map_err(|e| vm::HypervisorVmError::RegisterIrqFd(e.into()));
        log::debug!("Themis register_irqfd gsi={gsi} vec={vector} result={r:?}");
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
        // Push the resolved MSR interposition policy BEFORE the first
        // THHV_CREATE_VP: the capavisor builds the VMCS MSR bitmap inside
        // `do_add_vp` as a pure projection of MsrPolicy. If the policy
        // hasn't been pushed yet, the engine's restricted-child default
        // (Trap) is projected — every MSR access exits, including EFER
        // writes needed to enter long mode, and boot triple-faults.
        // Idempotent: subsequent calls are cheap no-ops.
        self.state
            .ensure_msr_policy_pushed()
            .map_err(vm::HypervisorVmError::CreateVcpu)?;

        let meta =
            MmapRegion::new_shared_anonymous(self.state.vp_meta_pages * self.state.page_size)
                .map_err(vm::HypervisorVmError::CreateVcpu)?;
        let comm = MmapRegion::new_shared_anonymous(self.state.page_size)
            .map_err(vm::HypervisorVmError::CreateVcpu)?;
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

        let timer = super::timer::LapicTimer::new(self.state.fd.clone(), id)
            .map_err(vm::HypervisorVmError::CreateVcpu)?;

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

        // Legacy APs wait for parent-emulated SIPI. PV APs enter private
        // guest firmware with state established before sealing instead.
        if id > 0 && !self.uses_guest_mp_wakeup() {
            let regs = [reg(VpRegister::ActivityState, 3)];
            let mut regs = regs.to_vec();
            let mut header = ThhvVpRegisters {
                count: regs.len() as u32,
                rsvd: 0,
                regs: regs.as_mut_ptr() as usize as u64,
            };
            ioctl_with_mut_ref(vcpu_fd.as_raw_fd(), THHV_SET_VP_STATE, &mut header)
                .map_err(|e| vm::HypervisorVmError::CreateVcpu(e.into()))?;
            log::debug!("Themis vCPU {id}: waiting for SIPI");
        }

        // Initialize per-vCPU software LAPIC state.
        {
            let mut lapic = self.state.lapic_regs.lock().unwrap();
            while lapic.len() <= id as usize {
                lapic.push([0u32; 1024]);
            }
            let regs = &mut lapic[id as usize];
            regs[0x020 / 4] = id << 24; // APIC ID
            regs[0x030 / 4] = 0x0005_0014; // Version: v20, 6 LVT entries
            regs[0x0D0 / 4] = 0; // LDR
            regs[0x0E0 / 4] = 0xFFFF_FFFF; // DFR: flat model
            regs[0x0F0 / 4] = 0x0000_01FF; // SVR: APIC enabled, vector 0xFF
            // LVT entries: masked by default
            regs[0x320 / 4] = 0x0001_0000; // LVT Timer (masked)
            regs[0x350 / 4] = 0x0001_0000; // LVT LINT0 (masked)
            regs[0x360 / 4] = 0x0001_0000; // LVT LINT1 (masked)
            regs[0x370 / 4] = 0x0001_0000; // LVT Error (masked)
            regs[0x340 / 4] = 0x0001_0000; // LVT PerfMon (masked)
            regs[0x330 / 4] = 0x0001_0000; // LVT Thermal (masked)
            regs[0x3E0 / 4] = 0x0000_000B; // Timer DCR: divide by 1
        }

        Ok(Box::new(ThemisVcpu {
            fd: vcpu_fd,
            _vp_index: id,
            vm_state: self.state.clone(),
            vm_ops,
            _meta: meta,
            _comm: comm,
            cpuid: std::sync::Mutex::new(Vec::new()),
            timer,
            msr_shadow: std::sync::Mutex::new(std::collections::HashMap::new()),
        }))
    }

    fn register_ioevent(
        &self,
        fd: &EventFd,
        addr: &IoEventAddress,
        datamatch: Option<DataMatch>,
    ) -> vm::Result<()> {
        self.state
            .ioevents
            .lock()
            .unwrap()
            .register(fd, *addr, datamatch)
            .map_err(|e| vm::HypervisorVmError::RegisterIoEvent(e.into()))
    }

    fn unregister_ioevent(&self, fd: &EventFd, addr: &IoEventAddress) -> vm::Result<()> {
        self.state
            .ioevents
            .lock()
            .unwrap()
            .unregister(fd.as_raw_fd(), *addr)
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
            match entry {
                IrqRoutingEntry::Themis(e) => {
                    if e.is_msi {
                        let vector = (e.msi_data & ICR_VECTOR_MASK) as u8;
                        // MSI address bits [19:12] = destination APIC ID
                        let dest_apic = ((e.msi_address_lo >> MSI_ADDR_DEST_ID_SHIFT)
                            & MSI_ADDR_DEST_ID_MASK) as u8;
                        log::debug!(
                            "Themis GSI routing: gsi={} msi_vector={} dest_apic={}",
                            e.gsi,
                            vector,
                            dest_apic
                        );
                        map.insert(e.gsi, (vector, dest_apic));
                    }
                }
                // Other backends' routing entries are not produced when the VMM is
                // talking to the Themis hypervisor; skip them rather than failing
                // so multi-backend builds (e.g. themis+kvm) still compile.
                #[cfg(feature = "kvm")]
                IrqRoutingEntry::Kvm(_) => {}
                #[cfg(feature = "mshv")]
                IrqRoutingEntry::Mshv(_) => {}
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
        // In confidential mode, guest RAM uses CARVE (exclusive — dom0 loses
        // access) while MMIO regions (PCI hole 0xC0000000–0xFFFFFFFF) stay ALIAS.
        let is_mmio = (0xC000_0000..0x1_0000_0000).contains(&guest_phys_addr);
        let flags = if self.state.confidential && !is_mmio {
            0 // CARVE (no ALIAS flag)
        } else {
            THHV_MEM_F_ALIAS
        };
        let region = ThhvSetGuestMemory {
            guest_pfn: guest_phys_addr >> 12,
            userspace_addr: userspace_addr as usize as u64,
            size: memory_size as u64,
            flags,
            rights: THHV_MEM_R_READ | THHV_MEM_R_WRITE | THHV_MEM_R_EXEC,
            attrs: 0,
            ..ThhvSetGuestMemory::default()
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
            ..ThhvSetGuestMemory::default()
        };
        ioctl_with_mut_ref(
            self.state.fd.as_raw_fd(),
            THHV_SET_GUEST_MEMORY,
            &mut region,
        )
        .map_err(|e| vm::HypervisorVmError::RemoveUserMemory(e.into()))
    }

    fn enable_split_irq(&self) -> vm::Result<()> {
        // KVM-specific split-irqchip mode (LAPIC in kernel, IOAPIC/PIC in
        // userspace).  Themis routes everything through capavisor; nothing
        // to enable.
        Ok(())
    }

    fn get_clock(&self) -> vm::Result<crate::ClockData> {
        // Used for live-migration save/restore.  Themis does not support
        // migration.
        Err(vm::HypervisorVmError::GetClock(anyhow!("not supported")))
    }

    fn set_clock(&self, _data: &crate::ClockData) -> vm::Result<()> {
        // See get_clock.
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

    fn annotate_shmem(&self, guest_gpa: u64, mode: u32, count: u32, path: &str) -> vm::Result<()> {
        let guest_pfn = guest_gpa >> 12;
        let mut pending = self.state.pending_memory.lock().unwrap();
        let entry = pending.iter_mut().find(|e| e.guest_pfn == guest_pfn);
        match entry {
            Some(e) => {
                e.shmem_mode = mode;
                e.shmem_count = count;
                // shmem_mode=ALIAS means dom0 keeps access → primary op is ALIAS.
                if mode == shmem_mode::ALIAS {
                    e.flags |= THHV_MEM_F_ALIAS;
                }
                let path_bytes = path.as_bytes();
                let copy_len = path_bytes.len().min(THHV_SHMEM_PATH_MAX - 1);
                e.shmem_path[..copy_len].copy_from_slice(&path_bytes[..copy_len]);
                log::debug!(
                    "Themis annotated shmem gfn=0x{guest_pfn:x} path=\"{path}\" mode={mode} count={count}"
                );
                Ok(())
            }
            None => {
                log::warn!("Themis annotate_shmem: no pending entry for gfn=0x{guest_pfn:x}");
                Err(vm::HypervisorVmError::CreateUserMemory(anyhow::anyhow!(
                    "no pending entry for shmem annotation"
                )))
            }
        }
    }

    fn register_ivshmem_bars(
        &self,
        index: u32,
        bar0_gpa: u64,
        bar2_gpa: u64,
        _count: u32,
    ) -> vm::Result<()> {
        let mut bars = self.state.ivshmem_bars.lock().unwrap();
        bars.push((index, bar0_gpa, bar2_gpa));
        log::debug!(
            "Themis registered ivshmem[{index}] bar0=0x{bar0_gpa:x} bar2=0x{bar2_gpa:x} (total={})",
            bars.len()
        );
        Ok(())
    }
}
