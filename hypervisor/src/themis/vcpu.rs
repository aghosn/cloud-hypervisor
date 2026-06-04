//! `ThemisVcpu`: per-vCPU state + the `Vcpu` trait implementation for the
//! Themis backend.
//!
//! Most state is owned by capavisor (the bare-metal hypervisor); methods that
//! would write hardware state on KVM/MSHV are no-ops here and carry a
//! `// Owned by capavisor:` comment.  Methods that legitimately need to alter
//! guest state go through THHV ioctls (THHV_SET_VP_STATE, THHV_GET_VP_STATE,
//! THHV_RUN_VP, …) so that capability checks happen in the kernel module.

use std::os::fd::{AsRawFd, OwnedFd};
use std::ptr;
use std::sync::Arc;

use anyhow::anyhow;
use vmm_sys_util::eventfd::EventFd;

use super::abi::{
    ThemicInterceptMessage, ThemisStandardRegisters, ThhvInjectInterrupt, ThhvRegNameValue,
    ThhvRunVp, ThhvVpRegisters, THHV_GET_VP_STATE, THHV_INJECT_INTERRUPT, THHV_RUN_VP,
    THHV_SET_VP_STATE,
};
use super::consts::{
    APIC_REG_EOI, APIC_REG_ICR_HIGH, APIC_REG_ICR_LOW, EMPTY_BOOT_MSRS, EPT_VIOLATION_EXECUTE,
    EXIT_REASON_APIC_ACCESS, EXIT_REASON_CPUID, EXIT_REASON_CR_ACCESS, EXIT_REASON_EPT_VIOLATION,
    EXIT_REASON_EXCEPTION_NMI, EXIT_REASON_EXTERNAL_INTERRUPT, EXIT_REASON_HLT,
    EXIT_REASON_IO_INSTRUCTION, EXIT_REASON_RDMSR, EXIT_REASON_TRIPLE_FAULT, EXIT_REASON_VMCALL,
    EXIT_REASON_WRMSR, ICR_DELIVERY_MODE_MASK, ICR_DELIVERY_MODE_SHIFT, ICR_DEST_SHORTHAND_MASK,
    ICR_DEST_SHORTHAND_SHIFT, ICR_HIGH_DEST_MASK, ICR_HIGH_DEST_SHIFT, ICR_MODE_FIXED,
    ICR_MODE_INIT, ICR_MODE_LOWEST_PRIORITY, ICR_MODE_SIPI, ICR_VECTOR_MASK, REALMODE_CODE_SEG_AR,
    REALMODE_DATA_SEG_AR, STANDARD_REGS, THEMIC_MSG_SHUTDOWN, THEMIS_EXIT_DOORBELL, VpRegister,
};
use super::emulator;
use super::helpers::{ioctl_with_mut_ref, reg, segment_from_raw, segment_regs};
use super::mmap::MmapRegion;
use super::vm_state::ThemisVmState;
use crate::arch::x86::{CpuIdEntry, FpuState, LapicState, MsrEntry, SpecialRegisters};
use crate::cpu::{self, HypervisorCpuError, Vcpu, VmExit};
use crate::vm::VmOps;
use crate::{CpuState, MpState};

pub struct ThemisVcpu {
    pub(super) fd: OwnedFd,
    pub(super) _vp_index: u32,
    pub(super) vm_state: Arc<ThemisVmState>,
    pub(super) vm_ops: Option<Arc<dyn VmOps>>,
    pub(super) _meta: MmapRegion,
    pub(super) _comm: MmapRegion,
    /// CPUID policy set by CHV via set_cpuid2() before the first run.
    /// Searched by (function, index) on every CPUID exit.
    pub(super) cpuid: std::sync::Mutex<Vec<CpuIdEntry>>,
    /// LAPIC timer emulation: timerfd armed to the TSC deadline.
    /// A background thread reads the timerfd and signals `timer_irq`.
    pub(super) timer_fd: OwnedFd,
    /// EventFd registered as irqfd for LOCAL_TIMER_VECTOR (0xEC).
    /// Signalled by the timer thread when the deadline fires.
    pub(super) _timer_irq: EventFd,
}

impl ThemisVcpu {
    /// Access the VmOps callbacks (MMIO/PIO/guest-mem dispatch).
    pub fn vm_ops(&self) -> Option<&Arc<dyn VmOps>> {
        self.vm_ops.as_ref()
    }
}



impl Vcpu for ThemisVcpu {
    fn create_standard_regs(&self) -> crate::StandardRegisters {
        crate::StandardRegisters::Themis(ThemisStandardRegisters::default())
    }

    fn get_regs(&self) -> cpu::Result<crate::StandardRegisters> {
        let values = self.get_reg_values(&STANDARD_REGS)?;
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
            reg(VpRegister::Rax, regs.rax),
            reg(VpRegister::Rbx, regs.rbx),
            reg(VpRegister::Rcx, regs.rcx),
            reg(VpRegister::Rdx, regs.rdx),
            reg(VpRegister::Rsi, regs.rsi),
            reg(VpRegister::Rdi, regs.rdi),
            reg(VpRegister::Rsp, regs.rsp),
            reg(VpRegister::Rbp, regs.rbp),
            reg(VpRegister::R8, regs.r8),
            reg(VpRegister::R9, regs.r9),
            reg(VpRegister::R10, regs.r10),
            reg(VpRegister::R11, regs.r11),
            reg(VpRegister::R12, regs.r12),
            reg(VpRegister::R13, regs.r13),
            reg(VpRegister::R14, regs.r14),
            reg(VpRegister::R15, regs.r15),
            reg(VpRegister::Rip, regs.rip),
            reg(VpRegister::Rflags, regs.rflags),
        ];
        let result = self.set_reg_values(&regs)
            .map_err(|e| HypervisorCpuError::SetStandardRegs(anyhow!(e.to_string())));
        if let Err(ref e) = result {
            eprintln!("\r[THEMIS-DBG] set_regs: THHV_SET_VP_STATE FAILED: {e}");
        }
        result
    }

    fn get_sregs(&self) -> cpu::Result<SpecialRegisters> {
        let names = [
            VpRegister::Cr0,
            VpRegister::Cr3,
            VpRegister::Cr4,
            VpRegister::Efer,
            VpRegister::ApicBase,
            VpRegister::CsSelector,
            VpRegister::CsBase,
            VpRegister::CsLimit,
            VpRegister::CsAccessRights,
            VpRegister::DsSelector,
            VpRegister::DsBase,
            VpRegister::DsLimit,
            VpRegister::DsAccessRights,
            VpRegister::EsSelector,
            VpRegister::EsBase,
            VpRegister::EsLimit,
            VpRegister::EsAccessRights,
            VpRegister::FsSelector,
            VpRegister::FsBase,
            VpRegister::FsLimit,
            VpRegister::FsAccessRights,
            VpRegister::GsSelector,
            VpRegister::GsBase,
            VpRegister::GsLimit,
            VpRegister::GsAccessRights,
            VpRegister::SsSelector,
            VpRegister::SsBase,
            VpRegister::SsLimit,
            VpRegister::SsAccessRights,
            VpRegister::TrSelector,
            VpRegister::TrBase,
            VpRegister::TrLimit,
            VpRegister::TrAccessRights,
            VpRegister::LdtrSelector,
            VpRegister::LdtrBase,
            VpRegister::LdtrLimit,
            VpRegister::LdtrAccessRights,
            VpRegister::GdtrBase,
            VpRegister::GdtrLimit,
            VpRegister::IdtrBase,
            VpRegister::IdtrLimit,
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
            reg(VpRegister::Cr0, sregs.cr0),
            reg(VpRegister::Cr3, sregs.cr3),
            reg(VpRegister::Cr4, sregs.cr4),
            reg(VpRegister::Efer, sregs.efer),
            reg(VpRegister::ApicBase, sregs.apic_base),
        ]);
        regs.extend_from_slice(&segment_regs(
            VpRegister::CsSelector,
            VpRegister::CsBase,
            VpRegister::CsLimit,
            VpRegister::CsAccessRights,
            &sregs.cs,
        ));
        regs.extend_from_slice(&segment_regs(
            VpRegister::DsSelector,
            VpRegister::DsBase,
            VpRegister::DsLimit,
            VpRegister::DsAccessRights,
            &sregs.ds,
        ));
        regs.extend_from_slice(&segment_regs(
            VpRegister::EsSelector,
            VpRegister::EsBase,
            VpRegister::EsLimit,
            VpRegister::EsAccessRights,
            &sregs.es,
        ));
        regs.extend_from_slice(&segment_regs(
            VpRegister::FsSelector,
            VpRegister::FsBase,
            VpRegister::FsLimit,
            VpRegister::FsAccessRights,
            &sregs.fs,
        ));
        regs.extend_from_slice(&segment_regs(
            VpRegister::GsSelector,
            VpRegister::GsBase,
            VpRegister::GsLimit,
            VpRegister::GsAccessRights,
            &sregs.gs,
        ));
        regs.extend_from_slice(&segment_regs(
            VpRegister::SsSelector,
            VpRegister::SsBase,
            VpRegister::SsLimit,
            VpRegister::SsAccessRights,
            &sregs.ss,
        ));
        regs.extend_from_slice(&segment_regs(
            VpRegister::TrSelector,
            VpRegister::TrBase,
            VpRegister::TrLimit,
            VpRegister::TrAccessRights,
            &sregs.tr,
        ));
        regs.extend_from_slice(&segment_regs(
            VpRegister::LdtrSelector,
            VpRegister::LdtrBase,
            VpRegister::LdtrLimit,
            VpRegister::LdtrAccessRights,
            &sregs.ldt,
        ));
        regs.extend_from_slice(&[
            reg(VpRegister::GdtrBase, sregs.gdt.base),
            reg(VpRegister::GdtrLimit, sregs.gdt.limit as u64),
            reg(VpRegister::IdtrBase, sregs.idt.base),
            reg(VpRegister::IdtrLimit, sregs.idt.limit as u64),
        ]);
        self.set_reg_values(&regs)
            .map_err(|e| HypervisorCpuError::SetSpecialRegs(anyhow!(e.to_string())))
    }

    fn get_fpu(&self) -> cpu::Result<FpuState> {
        // Owned by capavisor: FPU state lives in the per-vCPU XSAVE area
        // controlled by capavisor's VMCS init.  CHV has no read path.
        Err(HypervisorCpuError::GetFloatingPointRegs(anyhow!(
            "not supported"
        )))
    }

    fn set_fpu(&self, _fpu: &FpuState) -> cpu::Result<()> {
        // Owned by capavisor: FCW/MXCSR defaults are programmed at VMCS init
        // time.  Silently swallow the CHV setup_fpu() call (see
        // arch/src/x86_64/regs.rs).
        Ok(())
    }

    fn set_cpuid2(&self, cpuid: &[CpuIdEntry]) -> cpu::Result<()> {
        // Store per-vCPU for trap-and-emulate fallback.
        let mut guard = self.cpuid.lock().unwrap();
        *guard = cpuid.to_vec();
        // Also store in VM state for interposition policy push (first wins).
        let mut vm_entries = self.vm_state.cpuid_entries.lock().unwrap();
        if vm_entries.is_empty() {
            *vm_entries = cpuid.to_vec();
        }
        Ok(())
    }

    fn enable_hyperv_synic(&self) -> cpu::Result<()> {
        // HyperV-specific; not supported.  Returning Ok is consistent with
        // mshv's stub for non-HyperV guests.
        Ok(())
    }

    fn get_cpuid2(&self, _num_entries: usize) -> cpu::Result<Vec<CpuIdEntry>> {
        // Owned by capavisor: guest CPUID is masked by capavisor on each
        // VMEXIT(CPUID).  CHV's set_cpuid2() is captured into vm_state for the
        // trap-and-emulate fallback only; readback from hardware is not
        // available.
        Ok(Vec::new())
    }

    fn get_lapic(&self) -> cpu::Result<LapicState> {
        // Owned by capavisor: LAPIC virtualization is in capavisor.  CHV uses
        // this only for save/restore (live migration), which Themis does not
        // support.
        Ok(LapicState::default())
    }

    fn set_lapic(&self, _lapic: &LapicState) -> cpu::Result<()> {
        // Owned by capavisor: see get_lapic.
        Ok(())
    }

    fn get_msrs(&self, _msrs: &mut Vec<MsrEntry>) -> cpu::Result<usize> {
        // Owned by capavisor: guest MSRs are managed via the VMCS MSR-load /
        // MSR-store areas configured by capavisor at VMCS init.  CHV has no
        // direct read path.  Pair with `boot_msr_entries() -> &EMPTY_BOOT_MSRS`
        // so that arch::x86_64::regs::setup_msrs() is a no-op.
        Ok(0)
    }

    fn set_msrs(&self, _msrs: &[MsrEntry]) -> cpu::Result<usize> {
        // Owned by capavisor: see get_msrs.
        Ok(0)
    }

    fn get_mp_state(&self) -> cpu::Result<MpState> {
        // MP state is internal to capavisor (INIT/SIPI handled there); we
        // expose a Themis-flavored MpState variant so CHV's state machine
        // round-trips without the value being meaningful.
        Ok(MpState::Themis)
    }

    fn set_mp_state(&self, _mp_state: MpState) -> cpu::Result<()> {
        // Owned by capavisor: see get_mp_state.
        Ok(())
    }

    fn tsc_khz(&self) -> cpu::Result<Option<u32>> {
        // Try CPUID leaf 0x15: TSC/Crystal ratio (Intel SDM Vol. 3A §18.7.3).
        // EBX/EAX is the TSC-to-crystal multiplier/denominator; ECX is the
        // crystal frequency in Hz.  If ECX is 0 (common in older models) we
        // fall back to the Tiger/Ice-Lake nominal 19.2 MHz crystal.
        // CPUID is always available on x86_64.
        let leaf15 = std::arch::x86_64::__cpuid(0x15);
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
        let leaf16 = std::arch::x86_64::__cpuid(0x16);
        if (leaf16.eax & 0xffff) != 0 {
            return Ok(Some((leaf16.eax & 0xffff) * 1000));
        }

        Ok(None)
    }

    fn state(&self) -> cpu::Result<CpuState> {
        // vCPU state save/restore is for live migration; not supported.
        Err(HypervisorCpuError::GetCpuid(anyhow!(
            "state save not supported"
        )))
    }

    fn set_state(&self, _state: &CpuState) -> cpu::Result<()> {
        // See state().
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
        // Pair with set_msrs/get_msrs no-ops: capavisor sets boot MSRs at
        // VMCS init time, so we tell CHV there is nothing to push.
        &EMPTY_BOOT_MSRS
    }

    fn nmi(&self) -> cpu::Result<()> {
        // NMI injection is part of capavisor's interrupt model; CHV cannot
        // raise NMIs into a guest directly.
        Err(HypervisorCpuError::Nmi(anyhow!("not supported")))
    }
}

impl ThemisVcpu {
    fn get_reg_values(&self, names: &[VpRegister]) -> cpu::Result<Vec<u64>> {
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

        // Temporary: log first few exits for visibility (no AP-specific logging today).
        static EXIT_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let en = EXIT_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if en < 5 {
            eprintln!(
                "\r[THEMIS-EXIT] #{en} vp={} reason={} rip={:#x} msr_num={:#x}",
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
                let cr0  = self.get_reg_values(&[VpRegister::Cr0]).unwrap_or_default();
                let cr3  = self.get_reg_values(&[VpRegister::Cr3]).unwrap_or_default();
                let cr4  = self.get_reg_values(&[VpRegister::Cr4]).unwrap_or_default();
                let efer = self.get_reg_values(&[VpRegister::Efer]).unwrap_or_default();
                let rax  = self.get_reg_values(&[VpRegister::Rax]).unwrap_or_default();
                eprintln!("\r[TRIPLE-FAULT] rip={:#x} cr0={:#x} cr3={:#x} cr4={:#x} efer={:#x} rax={:#x}",
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
            THEMIS_EXIT_DOORBELL => Ok(VmExit::Ignore),
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
                "\r[THEMIS-MSR] #{n} WRMSR msr={:#x} val={:#x} rip={:#x}",
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
                "\r[THEMIS-TIMER] #{n} deadline={:#x} now={:#x} delta_tsc={} ({} us)",
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
        let thhv_gpr = |idx: u32| -> VpRegister {
            match idx {
                0  => VpRegister::Rax,
                1  => VpRegister::Rcx,
                2  => VpRegister::Rdx,
                3  => VpRegister::Rbx,
                4  => VpRegister::Rsp,
                5  => VpRegister::Rbp,
                6  => VpRegister::Rsi,
                7  => VpRegister::Rdi,
                8  => VpRegister::R8,
                9  => VpRegister::R9,
                10 => VpRegister::R10,
                11 => VpRegister::R11,
                12 => VpRegister::R12,
                13 => VpRegister::R13,
                14 => VpRegister::R14,
                15 => VpRegister::R15,
                _  => VpRegister::Rax,
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
                    let old_cr0 = self.get_reg_values(&[VpRegister::Cr0])?[0];
                    // Keep bits that the host must own (in FIXED0 mask) that guest wants to clear.
                    let forced_bits = old_cr0 & !src_val;
                    let new_cr0 = src_val | forced_bits;
                    let pg_bit  = 1u64 << 31;
                    let mut updates = vec![reg(VpRegister::Cr0, new_cr0)];
                    if (old_cr0 & pg_bit) == 0 && (new_cr0 & pg_bit) != 0 {
                        let efer = self.get_reg_values(&[VpRegister::Efer])?[0];
                        if efer & (1 << 8) != 0 {
                            // LME set → activate LMA now that PG is going on.
                            updates.push(reg(VpRegister::Efer, efer | (1 << 10)));
                        }
                    }
                    self.set_reg_values(&updates)?;
                }
                3 => {
                    self.set_reg_values(&[reg(VpRegister::Cr3, src_val)])?;
                }
                4 => {
                    // Preserve VMXE (bit 13) — guest cannot clear it.
                    self.set_reg_values(&[reg(VpRegister::Cr4, src_val | (1u64 << 13))])?;
                }
                8 => { /* CR8 / TPR — ignore */ }
                _ => {}
            }
        } else if acc == 1 {
            // MOV from CR: read CR value and write to the destination register.
            let cr_reg = match cr_num {
                0 => Some(VpRegister::Cr0),
                3 => Some(VpRegister::Cr3),
                4 => Some(VpRegister::Cr4),
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
                // CPUID is always available on x86_64 and has no side effects.
                let r = std::arch::x86_64::__cpuid_count(leaf, subleaf);
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
            reg(VpRegister::Rax, u64::from(eax)),
            reg(VpRegister::Rbx, u64::from(ebx)),
            reg(VpRegister::Rcx, u64::from(ecx)),
            reg(VpRegister::Rdx, u64::from(edx)),
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
                "\r[THEMIS-IO] #{} port=0x{:x} size={} write={} rax=0x{:x}",
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
        self.set_reg_values(&[reg(VpRegister::Rax, eax as u64)])
    }

    fn handle_mmio_exit(&mut self, msg: &ThemicInterceptMessage) -> cpu::Result<()> {
        // Strip the VTOM bit from the GPA. CoCo guests set this bit on MMIO
        // addresses to mark them as shared.  The device model only knows the
        // canonical (non-VTOM'd) addresses.  Skip when not confidential.
        let gpa = if self.vm_state.confidential {
            let vtom_mask = 1u64 << self.vm_state.vtom_bit;
            msg.guest_physical_address & !vtom_mask
        } else {
            msg.guest_physical_address
        };

        // LAPIC MMIO fast-path: when VIRTUALIZE_APIC_ACCESSES is not
        // supported, LAPIC accesses cause EPT violations.  Handle them
        // here with per-vCPU software LAPIC state + instruction decode.
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
            mmio_gpa: gpa,
            insn_bytes: msg.instruction_bytes,
            vtom_mask: if self.vm_state.confidential {
                1u64 << self.vm_state.vtom_bit
            } else {
                0
            },
        };

        let old_rip = msg.guest_rip;

        let mut emu = crate::arch::x86::emulator::Emulator::new(&mut ctx);
        let new_state = emu
            .emulate_first_insn(0, &msg.instruction_bytes)
            .map_err(|e| {
                eprintln!(
                    "\r[MMIO-EMU] emulation failed: gpa={:#x} rip={:#x} insn={:02x?}: {e}",
                    gpa, msg.guest_rip, &msg.instruction_bytes[..8],
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
                "\r[MMIO-DBG] #{mn} gpa={:#x} old_rip={:#x} new_rip={:#x} insn={:02x?}",
                gpa, old_rip, new_rip, &msg.instruction_bytes[..6],
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
                "\r[LAPIC] failed to decode insn at RIP={:#x} bytes={:02x?}",
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
                            "\r[LAPIC] unhandled write operand kind {:?} at RIP={:#x}",
                            insn.op1_kind(),
                            msg.guest_rip
                        );
                        0
                    }
                }
            } else {
                eprintln!(
                    "\r[LAPIC] write with {} operands at RIP={:#x}",
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
                APIC_REG_ICR_LOW => {
                    // ICR low write — detect INIT/SIPI.
                    // Read ICR_HIGH from internal LAPIC state.
                    let icr_high = {
                        let lapic = self.vm_state.lapic_regs.lock().unwrap();
                        let vp = self._vp_index as usize;
                        if vp < lapic.len() { lapic[vp][(APIC_REG_ICR_HIGH / 4) as usize] } else { 0 }
                    };
                    self.deliver_ipi(value, icr_high);
                }
                APIC_REG_EOI => {
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
                    "\r[LAPIC-W] #{n} vp={} offset={:#x} value={:#x}",
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
                    "\r[LAPIC-R] #{n} vp={} offset={:#x} value={:#x}",
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
                eprintln!("\r[LAPIC] read_iced_reg: unhandled register {:?}", r);
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
                eprintln!("\r[LAPIC] write_iced_reg: unhandled register {:?}", r);
            }
        }
    }

    /// Deliver an IPI based on the ICR low value.
    /// Handles IPI delivery modes: Fixed (0), INIT (5), and SIPI (6).
    fn deliver_ipi(&self, icr_low: u32, icr_high: u32) {
        let delivery_mode = (icr_low >> ICR_DELIVERY_MODE_SHIFT) & ICR_DELIVERY_MODE_MASK;
        let vector = icr_low & ICR_VECTOR_MASK;
        let dest_shorthand = (icr_low >> ICR_DEST_SHORTHAND_SHIFT) & ICR_DEST_SHORTHAND_MASK;

        // Destination APIC ID from ICR_HIGH bits [31:24], passed by capavisor
        // from VAPIC[0x310].
        let dest_apic_id = (icr_high >> ICR_HIGH_DEST_SHIFT) & ICR_HIGH_DEST_MASK;

        eprintln!(
            "\r[LAPIC-IPI] vp={} icr_low={:#x} mode={} vector={:#x} shorthand={} dest_apic={}",
            self._vp_index, icr_low, delivery_mode, vector, dest_shorthand, dest_apic_id
        );

        match delivery_mode {
            ICR_MODE_FIXED | ICR_MODE_LOWEST_PRIORITY => {
                // Fixed (0) or Lowest Priority (1) — inject vector into target VP.
                self.deliver_fixed_ipi(vector as u8, dest_shorthand, dest_apic_id);
            }
            ICR_MODE_INIT => {
                // INIT — AP is already in wait-for-SIPI (set at create_vcpu).
                eprintln!("\r[LAPIC-IPI] INIT IPI — no-op (AP in wait-for-SIPI)");
            }
            ICR_MODE_SIPI => {
                // Startup IPI (SIPI) — wake target AP with CS:IP from vector.
                self.deliver_sipi(vector);
            }
            _ => {
                eprintln!(
                    "\r[LAPIC-IPI] unhandled delivery mode {} — ignored",
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
                    "\r[LAPIC-IPI] Fixed IPI inject failed: vp={} vector={:#x} err={:?}",
                    target_vp, vector, err
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

            // Intel SDM §8.4.4: SIPI vector × 4 KiB = real-mode entry address
            let startup_addr = (vector as u64) << 12;
            // Intel SDM §8.4.4: CS selector = SIPI vector × 256
            let cs_selector = (vector as u64) << 8;

            let regs = [
                // CS from SIPI vector
                reg(VpRegister::CsSelector, cs_selector),
                reg(VpRegister::CsBase, startup_addr),
                reg(VpRegister::CsLimit, 0xFFFF),
                reg(VpRegister::CsAccessRights, REALMODE_CODE_SEG_AR.into()),
                // DS/ES/FS/GS/SS = 0:0 with 64K limit (real-mode reset)
                reg(VpRegister::DsSelector, 0),
                reg(VpRegister::DsBase, 0),
                reg(VpRegister::DsLimit, 0xFFFF),
                reg(VpRegister::DsAccessRights, REALMODE_DATA_SEG_AR.into()),
                reg(VpRegister::EsSelector, 0),
                reg(VpRegister::EsBase, 0),
                reg(VpRegister::EsLimit, 0xFFFF),
                reg(VpRegister::EsAccessRights, REALMODE_DATA_SEG_AR.into()),
                reg(VpRegister::FsSelector, 0),
                reg(VpRegister::FsBase, 0),
                reg(VpRegister::FsLimit, 0xFFFF),
                reg(VpRegister::FsAccessRights, REALMODE_DATA_SEG_AR.into()),
                reg(VpRegister::GsSelector, 0),
                reg(VpRegister::GsBase, 0),
                reg(VpRegister::GsLimit, 0xFFFF),
                reg(VpRegister::GsAccessRights, REALMODE_DATA_SEG_AR.into()),
                reg(VpRegister::SsSelector, 0),
                reg(VpRegister::SsBase, 0),
                reg(VpRegister::SsLimit, 0xFFFF),
                reg(VpRegister::SsAccessRights, REALMODE_DATA_SEG_AR.into()),
                // GDT/IDT with zero base (Linux trampoline sets its own)
                reg(VpRegister::GdtrBase, 0),
                reg(VpRegister::GdtrLimit, 0xFFFF),
                reg(VpRegister::IdtrBase, 0),
                reg(VpRegister::IdtrLimit, 0xFFFF),
                // RIP = 0 (offset within CS segment)
                reg(VpRegister::Rip, 0),
                // RFLAGS = 0x2 (reserved bit 1 always set)
                reg(VpRegister::Rflags, 0x2),
                // CR0: PE=0 for real mode. The capavisor's vmcs_adjust_cr0()
                // will add NE/ET bits required by IA32_VMX_CR0_FIXED0.
                // UNRESTRICTED_GUEST exempts PE and PG from FIXED0 requirements.
                reg(VpRegister::Cr0, 0x00),
                reg(VpRegister::Cr3, 0),
                reg(VpRegister::Cr4, 0),
                reg(VpRegister::Efer, 0),
                // Clear GP regs
                reg(VpRegister::Rax, 0),
                reg(VpRegister::Rbx, 0),
                reg(VpRegister::Rcx, 0),
                reg(VpRegister::Rdx, 0),
                reg(VpRegister::Rsi, 0),
                reg(VpRegister::Rdi, 0),
                reg(VpRegister::Rbp, 0),
                reg(VpRegister::Rsp, 0),
                // Wake the AP
                reg(VpRegister::ActivityState, 0),
            ];
            let mut regs = regs.to_vec();
            let mut header = ThhvVpRegisters {
                count: regs.len() as u32,
                rsvd: 0,
                regs: regs.as_mut_ptr() as usize as u64,
            };
            let res = ioctl_with_mut_ref(target_fd, THHV_SET_VP_STATE, &mut header);
            eprintln!(
                "\r[LAPIC-IPI] SIPI vector={:#x} → vCPU {} CS:IP={:#x}:{:#x} result={:?}",
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
