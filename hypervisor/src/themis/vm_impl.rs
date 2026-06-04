//! `ThemisVm` and its `vm::Vm` trait implementation.
//!
//! Owns the per-VM `Arc<ThemisVmState>` and translates CHV's `vm::Vm`
//! operations into THHV ioctls (memory regions, irqfd, ioeventfd, vCPU
//! creation, GSI routing).  Stub methods that are no-ops in the Themis
//! trust model carry `// Owned by capavisor:` comments.

use std::any::Any;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::Arc;

use anyhow::anyhow;
use libc::c_void;
use vfio_ioctls::VfioDeviceFd;
use vmm_sys_util::eventfd::EventFd;

use super::abi::*;
use super::consts::*;
use super::helpers::{ioctl_with_mut_ref, ioctl_with_mut_ref_ret_fd, reg};
use super::mmap::MmapRegion;
use super::vm_state::ThemisVmState;
use super::ThemisVcpu;
use crate::cpu::Vcpu;
use crate::vm::{self, DataMatch, InterruptSourceConfig, VmOps};
use crate::{IoEventAddress, IrqRoutingEntry};

pub struct ThemisVm {
    pub(super) state: Arc<ThemisVmState>,
}

impl vm::Vm for ThemisVm {
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
        let (vector, dest_apic) = self.state.gsi_vectors.lock().unwrap()
            .get(&gsi).copied().unwrap_or((0, 0));
        let vector = vector as u32;
        eprintln!("\r[THEMIS-DBG] register_irqfd gsi={gsi} vec={vector} vp={dest_apic} fd={}", fd.as_raw_fd());
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
        eprintln!("\r[THEMIS-DBG] register_irqfd gsi={gsi} vec={vector} result={r:?}");
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
            "\r[THEMIS-DBG] vCPU {id}: LAPIC timer emulation ready (timerfd={}, irqfd vec=0x{LOCAL_TIMER_VECTOR:X})",
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
            let regs = [reg(VpRegister::ActivityState, 3)];
            let mut regs = regs.to_vec();
            let mut header = ThhvVpRegisters {
                count: regs.len() as u32,
                rsvd: 0,
                regs: regs.as_mut_ptr() as usize as u64,
            };
            ioctl_with_mut_ref(vcpu_fd.as_raw_fd(), THHV_SET_VP_STATE, &mut header)
                .map_err(|e| vm::HypervisorVmError::CreateVcpu(e.into()))?;
            eprintln!("\r[THEMIS-DBG] vCPU {id}: set ACTIVITY_STATE=3 (wait-for-SIPI)");
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
        let ioevent = ThhvIoeventfd {
            fd: fd.as_raw_fd(),
            flags,
            addr,
            len,
            datamatch,
        };
        // If the domain is not yet sealed, defer the ioctl.  THHV_IOEVENTFD
        // calls REGISTER_DOORBELL which needs the child domain to exist.
        //
        // IMPORTANT: when deferring, we MUST store the cloned fd's number,
        // not the original.  The original `EventFd` belongs to the caller and
        // is typically dropped right after `register_ioevent` returns, which
        // closes its raw fd in the process.  By the time the deferred ioctl
        // runs (inside ensure_initialized()), the original fd number may have
        // been reassigned to a completely unrelated eventfd — and thhv would
        // then register the doorbell against the wrong eventfd_ctx (in
        // practice, this aliased onto `exit_evt`, so doorbell rings would
        // tear the VM down).  The cloned fd is held alive in
        // `pending_ioeventfds` so its number stays valid until flush.
        if !*self.state.initialized.lock().unwrap() {
            let fd_clone = fd.try_clone().map_err(|e|
                vm::HypervisorVmError::RegisterIoEvent(e.into()))?;
            let mut ioevent = ioevent;
            ioevent.fd = fd_clone.as_raw_fd();
            eprintln!(
                "\r[THEMIS-DBG] deferring IOEVENTFD addr=0x{:x} flags=0x{:x}",
                ioevent.addr, ioevent.flags
            );
            self.state.pending_ioeventfds.lock().unwrap().push((ioevent, fd_clone));
            return Ok(());
        }
        let mut ioevent = ioevent;
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
            match entry {
                IrqRoutingEntry::Themis(e) => {
                    if e.is_msi {
                        let vector = (e.msi_data & ICR_VECTOR_MASK) as u8;
                        // MSI address bits [19:12] = destination APIC ID
                        let dest_apic = ((e.msi_address_lo >> MSI_ADDR_DEST_ID_SHIFT) & MSI_ADDR_DEST_ID_MASK) as u8;
                        eprintln!("\r[THEMIS-DBG] gsi_routing: gsi={} → msi_vector={} dest_apic={}", e.gsi, vector, dest_apic);
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
        let is_mmio = guest_phys_addr >= 0xC000_0000 && guest_phys_addr < 0x1_0000_0000;
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

    fn annotate_shmem(
        &self,
        guest_gpa: u64,
        mode: u32,
        count: u32,
        path: &str,
    ) -> vm::Result<()> {
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
                eprintln!(
                    "\r[THEMIS-DBG] annotated shmem gfn=0x{guest_pfn:x} path=\"{path}\" mode={mode} count={count}"
                );
                Ok(())
            }
            None => {
                eprintln!(
                    "\r[THEMIS-DBG] annotate_shmem: no pending entry for gfn=0x{guest_pfn:x}"
                );
                Err(vm::HypervisorVmError::CreateUserMemory(
                    anyhow::anyhow!("no pending entry for shmem annotation"),
                ))
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
        eprintln!(
            "\r[THEMIS-DBG] registered ivshmem[{index}] bar0=0x{bar0_gpa:x} bar2=0x{bar2_gpa:x} (total={})",
            bars.len()
        );
        Ok(())
    }
}
