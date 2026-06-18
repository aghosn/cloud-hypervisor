//! Per-VM state shared between `ThemisVm` and its `ThemisVcpu`s.
//!
//! Holds the THHV partition fd, mmap'd meta pages, deferred-init queues
//! (memory regions, ioeventfds), per-vCPU LAPIC software state, and the
//! interposition-policy bookkeeping pushed before sealing the partition.

use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use vmm_sys_util::eventfd::EventFd;

use super::abi::*;
use super::consts::*;
use super::helpers::{ioctl_noarg, ioctl_with_mut_ref};
use super::mmap::MmapRegion;
use crate::arch::x86::CpuIdEntry;

pub(crate) struct ThemisVmState {
    pub fd: Arc<OwnedFd>,
    pub initialized: Mutex<bool>,
    /// Whether the per-domain MSR interposition policy has been pushed to
    /// the capa engine. Pushed lazily but **before** the first
    /// `THHV_CREATE_VP`, because capavisor's `do_add_vp` builds the VMCS
    /// MSR bitmap as a pure projection of `MsrPolicy` (see
    /// `themis/capavisor/src/arch/x86_64/msr_bitmap.rs`). If the policy
    /// is not in place at that moment, the engine's restricted-child
    /// default (Trap) is projected, every MSR access exits, and the
    /// boot path corrupts. Idempotent: only the first call pushes.
    pub msr_policy_pushed: Mutex<bool>,
    pub page_size: usize,
    pub vp_meta_pages: usize,
    pub _shared_meta: MmapRegion,
    /// APIC-access sentinel page mapped at GPA 0xFEE00000 in the child domain.
    /// Kept alive here so the pinned page is not freed while the domain runs.
    pub _apic_access: MmapRegion,
    /// EventFd for the periodic timer interrupt injection.
    /// Kept alive so the IRQFd registration in thhv.ko remains valid.
    pub _timer_eventfd: Mutex<Option<EventFd>>,
    /// Guest memory regions recorded during create_user_memory_region but not
    /// yet sent to the domain.  THHV_SET_GUEST_MEMORY is deferred until
    /// ensure_initialized() so that cloud-hypervisor can finish writing
    /// firmware/ACPI/kernel data into the mmap'd pages while dom0 still holds
    /// EPT access.  Once SET_GUEST_MEMORY fires the capavisor removes those
    /// pages from dom0's EPT, so all writes must complete before then.
    pub pending_memory: Mutex<Vec<ThhvSetGuestMemory>>,
    /// GSI → (MSI vector, destination APIC ID) mapping.  Updated by
    /// set_gsi_routing() when the guest configures MSI Address/Data.
    pub gsi_vectors: Mutex<std::collections::HashMap<u32, (u8, u8)>>,
    /// Per-vCPU file descriptors (raw), indexed by vCPU ID.  Used for
    /// cross-vCPU state updates (e.g., BSP sending SIPI to an AP).
    pub vp_fds: Mutex<Vec<RawFd>>,
    /// Per-vCPU software LAPIC register state (1024 × u32 = 4 KiB per vCPU).
    /// Used when VIRTUALIZE_APIC_ACCESSES is not available: LAPIC MMIO
    /// accesses cause EPT violations that are emulated here in software.
    pub lapic_regs: Mutex<Vec<[u32; 1024]>>,
    /// CPUID entries to push as interposition Emulate policy before sealing.
    /// Populated by the first vCPU's set_cpuid2() call.
    pub cpuid_entries: Mutex<Vec<CpuIdEntry>>,
    /// ivshmem BAR addresses for CPUID leaf 0x40000004 (Emulate override).
    /// Each entry: (index, bar0_gpa, bar2_gpa).
    pub ivshmem_bars: Mutex<Vec<(u32, u64, u64)>>,
    /// IOEVENTFDs deferred until ensure_initialized().  The THHV_IOEVENTFD
    /// ioctl calls REGISTER_DOORBELL which needs the child domain to exist
    /// (after creation) but must happen before seal.
    pub pending_ioeventfds: Mutex<Vec<(ThhvIoeventfd, EventFd)>>,
    /// VTOM bit position (MAXPHYADDR - 1). Used to strip the VTOM bit from
    /// MMIO GPAs when the CoCo kernel marks device accesses as shared.
    pub vtom_bit: u32,
    /// Confidential mode: guest RAM is CARVEd (dom0 loses access) instead of
    /// ALIASed. MMIO regions remain ALIAS (shared by design).
    pub confidential: bool,
}

impl ThemisVmState {
    /// Known shared regions in a confidential guest.  These need ALIAS slots
    /// (parent retains access) instead of CARVE, plus a vTOM mirror ALIAS.
    const SHARED_REGIONS: &'static [(u64, u64)] = &[
        (0xA0000, 0x100000), // EBDA / legacy ACPI firmware area
    ];

    /// Split pending memory regions for confidential mode.
    ///
    /// Any region that overlaps a known shared area is split into up to 3 parts:
    ///   [region_start .. shared_start]  CARVE  (if non-empty)
    ///   [shared_start .. shared_end]    ALIAS  (shared)
    ///   [shared_end   .. region_end]    CARVE  (if non-empty)
    /// Plus a vTOM mirror ALIAS at (vtom_bit | shared_start).
    ///
    /// Non-overlapping regions pass through unchanged.
    pub(super) fn split_confidential_regions(
        regions: &[ThhvSetGuestMemory],
        vtom_bit: u32,
    ) -> Vec<ThhvSetGuestMemory> {
        let all_rights = THHV_MEM_R_READ | THHV_MEM_R_WRITE | THHV_MEM_R_EXEC;
        let mut out = Vec::new();

        for region in regions {
            let r_start = region.guest_pfn << 12;
            let r_end = r_start + region.size;

            // MMIO regions are always ALIAS, no splitting needed.
            if region.flags & THHV_MEM_F_ALIAS != 0 {
                out.push(*region);
                continue;
            }

            // Collect shared sub-ranges that overlap this region.
            let mut cuts: Vec<(u64, u64)> = Vec::new();
            for &(s_start, s_end) in Self::SHARED_REGIONS {
                let lo = s_start.max(r_start);
                let hi = s_end.min(r_end);
                if lo < hi {
                    cuts.push((lo, hi));
                }
            }

            if cuts.is_empty() {
                // No shared overlap — pass through as CARVE.
                out.push(*region);
                continue;
            }

            // Sort cuts by start address (already sorted for a single entry,
            // but future-proof for multiple shared regions).
            cuts.sort_by_key(|&(lo, _)| lo);

            // Walk through the region, emitting CARVE gaps and ALIAS shared parts.
            let mut cursor = r_start;
            for (s_lo, s_hi) in &cuts {
                // CARVE gap before this shared region.
                if cursor < *s_lo {
                    let gap_size = s_lo - cursor;
                    out.push(ThhvSetGuestMemory {
                        guest_pfn: cursor >> 12,
                        userspace_addr: region.userspace_addr + (cursor - r_start),
                        size: gap_size,
                        flags: 0,
                        rights: all_rights,
                        attrs: 0,
                        ..ThhvSetGuestMemory::default()
                    });
                }

                // ALIAS for the shared region.
                let shared_size = s_hi - s_lo;
                out.push(ThhvSetGuestMemory {
                    guest_pfn: s_lo >> 12,
                    userspace_addr: region.userspace_addr + (s_lo - r_start),
                    size: shared_size,
                    flags: THHV_MEM_F_ALIAS,
                    rights: all_rights,
                    attrs: 0,
                    ..ThhvSetGuestMemory::default()
                });

                // vTOM mirror ALIAS.
                let vtom_gpa = s_lo | (1u64 << vtom_bit);
                out.push(ThhvSetGuestMemory {
                    guest_pfn: vtom_gpa >> 12,
                    userspace_addr: region.userspace_addr + (s_lo - r_start),
                    size: shared_size,
                    flags: THHV_MEM_F_ALIAS,
                    rights: all_rights,
                    attrs: 0,
                    ..ThhvSetGuestMemory::default()
                });

                cursor = *s_hi;
            }

            // CARVE tail after last shared region.
            if cursor < r_end {
                let tail_size = r_end - cursor;
                out.push(ThhvSetGuestMemory {
                    guest_pfn: cursor >> 12,
                    userspace_addr: region.userspace_addr + (cursor - r_start),
                    size: tail_size,
                    flags: 0,
                    rights: all_rights,
                    attrs: 0,
                    ..ThhvSetGuestMemory::default()
                });
            }
        }

        out
    }

    /// Push the per-domain MSR interposition policy to the capa engine.
    ///
    /// **Must be called before the first `THHV_CREATE_VP`.** The capavisor
    /// builds the VMCS MSR bitmap inside `do_add_vp` as a pure projection
    /// of `MsrPolicy` (see `themis/capavisor/src/arch/x86_64/msr_bitmap.rs`).
    /// If this hasn't run by then, the engine's default for restricted
    /// children (`DefaultAction::Trap`, set by
    /// `capability_engine::domain::Domain::new_restricted`) gets projected
    /// into the bitmap — every MSR access exits, including FS_BASE /
    /// GS_BASE / SYSCALL MSRs that have VMCS shadow fields, and the
    /// passthrough handler in `vmexit/msr.rs` writes to physical MSRs
    /// without updating the VMCS shadows, corrupting the guest on the
    /// next VMENTRY.
    ///
    /// Idempotent: the `msr_policy_pushed` guard ensures the engine
    /// hypercalls only fire once per VM.
    pub(super) fn ensure_msr_policy_pushed(&self) -> anyhow::Result<()> {
        let mut pushed = self.msr_policy_pushed.lock().unwrap();
        if *pushed {
            return Ok(());
        }

        // Themis feature bit gating: capavisor advertises
        // FEATURE_X2APIC_VIRT (themis_abi::cpuid::feature_bits) when the
        // hardware exposes VIRT_X2APIC_MODE + APIC_REGISTER_VIRT + VID.
        // In nested envs (KVM-as-L0) these are typically absent — keep the
        // child in xAPIC mode, pushing only the TSC-deadline emulate.
        let x2apic_virt = themis_x2apic_virt_supported();

        // MSR_DEFAULT = Native: every MSR not explicitly overridden runs
        // natively (no VM exit). This matches the historical behavior of
        // dom1 (zeroed MSR bitmap at child seal). To harden a future
        // child, flip this to Trap and push explicit Native ranges for
        // the MSRs the guest is allowed to access without mediation.
        //
        // MSR_DEFAULT encoding: value = DefaultAction (0=Trap, 1=Native).
        self.set_policy(policy_kind::MSR_DEFAULT, 0, 0, 1)?;

        // IA32_TSC_DEADLINE (0x6E0): Emulate so the capavisor's internal
        // emulator handles WRMSR via the VMX preemption timer (no
        // userspace round-trip). The Emulate policy causes capavisor's
        // bitmap projection to trap both RDMSR and WRMSR for 0x6E0; the
        // capavisor's `msr_emulator` consumes the WRMSR. The stored value
        // is 0 (unused; writes are intercepted by the capavisor handler).
        //
        // MSR_EMULATE encoding: key = msr_number, sub_key = word_index,
        // value = u32 word.
        self.set_policy(policy_kind::MSR_EMULATE, 0x6E0, 0, 0)?;

        // x2APIC virtualization (Intel SDM Vol 3C §29.5):
        //
        //   • VMCS for child VMs sets VIRTUALIZE_X2APIC_MODE +
        //     APIC_REGISTER_VIRT + VID, with IA32_APIC_BASE pinned to
        //     EN|EXTD via the Emulate policy below.  Children boot
        //     directly in x2APIC mode — the xAPIC MMIO decoder in
        //     capavisor (handle_apic_access_exit) is never reached.
        //
        //   • Self-IPI WRMSR(0x83F) under VID=1 is fully hardware-handled
        //     (zero exits).  This is the primary motivation: it kills the
        //     LAPIC IPI flood from the TSC-deadline path.
        //
        //   • Cross-CPU IPI via ICR (MSR 0x830) must still exit because
        //     dispatch is software-routed.  We mark it Trap and forward
        //     to handle_wrmsr_exit → deliver_ipi.
        //
        //   • Skipped entirely when the host does not advertise
        //     FEATURE_X2APIC_VIRT (nested-KVM case): without hardware
        //     virtualization, pre-setting EXTD lets Linux switch to
        //     x2APIC ops and then #GP on the first native RDMSR 0x802.
        //
        // MSR_RANGE encoding: key = start_msr, sub_key = end_msr (inclusive),
        // value = DefaultAction (0=Trap, 1=Native).
        // Ranges must be non-overlapping, so split 0x800..=0x83F around 0x830.
        if x2apic_virt {
            self.set_policy(policy_kind::MSR_RANGE, 0x800, 0x82F, 1)?; // Native
            self.set_policy(policy_kind::MSR_RANGE, 0x830, 0x830, 0)?; // Trap (ICR)
            self.set_policy(policy_kind::MSR_RANGE, 0x831, 0x83F, 1)?; // Native

            // IA32_APIC_BASE (0x1B): Emulate.  Stored value advertises the
            // architectural LAPIC base with EN (bit 11) and EXTD (bit 10) set
            // — Linux sees x2APIC already enabled and skips the xAPIC→x2APIC
            // transition entirely.  WRMSR is consumed (no-op) by capavisor's
            // msr_emulator handler to prevent the guest from disabling EXTD.
            //
            //   bit 8  (BSP):  intentionally NOT set — Linux derives BSP from
            //                  the MADT, and the per-domain policy is the same
            //                  value for all vCPUs.
            //   bit 10 (EXTD): x2APIC mode enabled.
            //   bit 11 (EN):   xAPIC global enable (also required for x2APIC).
            //   bits 35:12     LAPIC base physical frame (0xFEE00 << 12).
            const IA32_APIC_BASE_VALUE: u64 = 0xFEE0_0000 | (1 << 11) | (1 << 10);
            self.set_policy(
                policy_kind::MSR_EMULATE,
                0x1B,
                0,
                IA32_APIC_BASE_VALUE & 0xFFFF_FFFF,
            )?;
            self.set_policy(
                policy_kind::MSR_EMULATE,
                0x1B,
                1,
                IA32_APIC_BASE_VALUE >> 32,
            )?;
        }

        eprintln!(
            "\r[THEMIS-DBG] pushed MSR policy: default=Native, Emulate(0x6E0{}), \
             x2apic_virt={}",
            if x2apic_virt { ",0x1B" } else { "" },
            x2apic_virt
        );
        *pushed = true;
        Ok(())
    }

    pub(super) fn ensure_initialized(&self) -> anyhow::Result<()> {
        let mut initialized = self.initialized.lock().unwrap();
        if *initialized {
            return Ok(());
        }

        // Flush all deferred SET_GUEST_MEMORY calls before sealing.
        // After this point dom0 loses EPT access to dom1's guest pages.
        let raw_regions: Vec<ThhvSetGuestMemory> =
            self.pending_memory.lock().unwrap().drain(..).collect();

        // In confidential mode, split regions around known shared areas (EBDA/ACPI).
        // Shared regions become ALIAS slots (+ vTOM mirror), the rest stays CARVE.
        let regions = if self.confidential {
            Self::split_confidential_regions(&raw_regions, self.vtom_bit)
        } else {
            raw_regions
        };

        eprintln!("\r[THEMIS-DBG] ensure_initialized: flushing {} memory regions", regions.len());
        for (i, mut region) in regions.iter().cloned().enumerate() {
            let kind = if region.flags & THHV_MEM_F_ALIAS != 0 { "ALIAS" } else { "CARVE" };
            let shmem_info = if region.shmem_mode != 0 {
                let path = std::str::from_utf8(&region.shmem_path)
                    .unwrap_or("<invalid>")
                    .trim_end_matches('\0');
                format!(" shmem_mode={} count={} path=\"{}\"",
                    region.shmem_mode, region.shmem_count, path)
            } else {
                String::new()
            };
            eprintln!("\r[THEMIS-DBG] SET_GUEST_MEMORY[{i}] gfn=0x{:x} size=0x{:x} {kind}{shmem_info}",
                region.guest_pfn, region.size);
            ioctl_with_mut_ref(self.fd.as_raw_fd(), THHV_SET_GUEST_MEMORY, &mut region)
                .context("deferred THHV_SET_GUEST_MEMORY failed")?;
            eprintln!("\r[THEMIS-DBG] SET_GUEST_MEMORY[{i}] done");
        }

        // Push CPUID entries as Emulate overrides before sealing.
        // Each entry becomes 2 set_policy calls (word 0 and word 1).
        // Word 0: value = (eax << 32) | ebx — creates the entry.
        // Word 1: value = (ecx << 32) | edx — updates remaining fields.
        // Key encodes (leaf << 32 | subleaf), sub_key = word index.
        //
        // Patch leaf 1 ECX bit 21 (X2APIC) to 1 unconditionally — children
        // are pinned to x2APIC mode by the VMCS controls (see
        // vmcs/controls.rs and the MSR_EMULATE(0x1B) policy above).  The
        // bit is normally set on modern Intel hosts, but the capa engine
        // is the only source of truth for the guest's CPUID, so we force
        // it here to keep the policy self-consistent independent of host
        // CPUID quirks.
        // Patch leaf 1 ECX bit 21 (X2APIC) to 1 only when capavisor
        // advertises FEATURE_X2APIC_VIRT and we therefore pushed the
        // x2APIC MSR policy in ensure_msr_policy_pushed.  Forcing the bit
        // without that backing policy would let the guest switch to
        // x2APIC ops with no hardware virtualization, #GP'ing on the
        // first RDMSR 0x802.
        let x2apic_virt = themis_x2apic_virt_supported();
        let mut cpuid_entries: Vec<CpuIdEntry> =
            self.cpuid_entries.lock().unwrap().clone();
        if x2apic_virt {
            for entry in cpuid_entries.iter_mut() {
                if entry.function == 1 && entry.index == 0 {
                    entry.ecx |= 1 << 21; // X2APIC
                }
            }
        }
        if !cpuid_entries.is_empty() {
            eprintln!(
                "\r[THEMIS-DBG] pushing {} CPUID entries as Emulate policy",
                cpuid_entries.len()
            );
            for entry in &cpuid_entries {
                let key = ((entry.function as u64) << 32) | (entry.index as u64);
                let word0 = ((entry.eax as u64) << 32) | (entry.ebx as u64);
                let word1 = ((entry.ecx as u64) << 32) | (entry.edx as u64);
                self.set_policy(policy_kind::CPUID_EMULATE, key, 0, word0)?;
                self.set_policy(policy_kind::CPUID_EMULATE, key, 1, word1)?;
            }
        }

        // Push ivshmem BAR addresses as Emulate CPUID leaf 0x40000004.
        // Each ivshmem device gets its own subleaf (index).
        // EAX = BAR0 GPA (32-bit), EBX = device count,
        // ECX = BAR2 GPA (low 32), EDX = BAR2 GPA (high 32).
        let ivshmem_bars = self.ivshmem_bars.lock().unwrap().clone();
        if !ivshmem_bars.is_empty() {
            let count = ivshmem_bars.len() as u32;
            eprintln!(
                "\r[THEMIS-DBG] pushing {} ivshmem CPUID entries (leaf 0x40000004)",
                count
            );
            for &(index, bar0_gpa, bar2_gpa) in &ivshmem_bars {
                let leaf: u64 = (0x40000004_u64 << 32) | (index as u64);
                let eax = bar0_gpa as u32;
                let ebx = count;
                let ecx = bar2_gpa as u32;
                let edx = (bar2_gpa >> 32) as u32;
                let word0 = ((eax as u64) << 32) | (ebx as u64);
                let word1 = ((ecx as u64) << 32) | (edx as u64);
                self.set_policy(policy_kind::CPUID_EMULATE, leaf, 0, word0)?;
                self.set_policy(policy_kind::CPUID_EMULATE, leaf, 1, word1)?;
                eprintln!(
                    "\r[THEMIS-DBG]   ivshmem[{index}]: bar0=0x{bar0_gpa:x} bar2=0x{bar2_gpa:x}"
                );
            }
        }

        // Push Native ranges for Themis hypervisor CPUID leaves so the child
        // can discover the capavisor.  The capavisor handles these natively
        // (signature, features, domcomm, limits, etc.).
        //
        // We must split the Native range around any Emulate leaves (ivshmem,
        // CoCo detection) because the engine rejects overlapping entries.
        // Collect Emulate leaf numbers, sort them, then emit Native segments
        // that skip those leaves.
        //
        // CPUID_RANGE encoding: key = (start_leaf << 32) | start_sub,
        //                       sub_key = (end_leaf << 32) | end_sub,
        //                       value = 1 (Native).
        {
            let range_start: u32 = 0x40000000;
            let range_end: u32 = 0x4FFFFFFF;

            // Gather all Emulate leaf numbers that fall in the Native range.
            let mut emulate_leaves: Vec<u32> = Vec::new();

            // ivshmem leaves (one per device, all at leaf 0x40000004).
            if !ivshmem_bars.is_empty() {
                emulate_leaves.push(0x40000004);
            }

            // CoCo detection leaf.
            if self.confidential {
                emulate_leaves.push(0x40000100);
            }

            emulate_leaves.sort();
            emulate_leaves.dedup();

            // Emit Native segments between emulate gaps.
            let mut cursor = range_start;
            for &leaf in &emulate_leaves {
                if cursor < leaf {
                    let s: u64 = (cursor as u64) << 32;
                    let e: u64 = ((leaf - 1) as u64) << 32 | 0xFFFFFFFF;
                    self.set_policy(policy_kind::CPUID_RANGE, s, e, 1)?;
                }
                cursor = leaf + 1;
            }
            if cursor <= range_end {
                let s: u64 = (cursor as u64) << 32;
                let e: u64 = (range_end as u64) << 32 | 0xFFFFFFFF;
                self.set_policy(policy_kind::CPUID_RANGE, s, e, 1)?;
            }

            eprintln!(
                "\r[THEMIS-DBG] pushed Native CPUID ranges (split around {} emulate leaves)",
                emulate_leaves.len()
            );
        }

        // Push CoCo detection Emulate leaf (0x40000100) if confidential.
        if self.confidential {
            let coco_leaf: u64 = 0x40000100_u64 << 32;
            let eax = self.vtom_bit;
            let ebx = u32::from_le_bytes(*b"Them");
            let ecx = u32::from_le_bytes(*b"isCo");
            let edx = u32::from_le_bytes(*b"Co\0\0");
            let word0 = ((eax as u64) << 32) | (ebx as u64);
            let word1 = ((ecx as u64) << 32) | (edx as u64);
            eprintln!("\r[THEMIS-DBG] pushing CoCo CPUID leaf 0x40000100: vtom_bit={}", self.vtom_bit);
            self.set_policy(policy_kind::CPUID_EMULATE, coco_leaf, 0, word0)?;
            self.set_policy(policy_kind::CPUID_EMULATE, coco_leaf, 1, word1)?;
        }

        // ── MSR interposition policy ───────────────────────────────────
        // Already pushed by `ensure_msr_policy_pushed()` before the first
        // `THHV_CREATE_VP`. Calling it again here is a cheap no-op (the
        // guard mutex returns immediately). Done this way because the
        // capavisor builds the MSR bitmap during `do_add_vp` as a
        // projection of `MsrPolicy`; the policy must be in place before
        // the first VP is created or the engine's restricted-child
        // default (Trap) is projected, every MSR exits, and boot dies.
        self.ensure_msr_policy_pushed()?;

        // Flush deferred IOEVENTFDs — registers doorbells with capavisor.
        // Must happen after domain creation but before seal.
        {
            let pending = self.pending_ioeventfds.lock().unwrap().drain(..).collect::<Vec<_>>();
            for (mut ioevent, _owner) in pending {
                eprintln!(
                    "\r[THEMIS-DBG] deferred IOEVENTFD addr=0x{:x} fd={}",
                    ioevent.addr, ioevent.fd
                );
                ioctl_with_mut_ref(self.fd.as_raw_fd(), THHV_IOEVENTFD, &mut ioevent)
                    .context("deferred THHV_IOEVENTFD failed")?;
            }
        }

        eprintln!("\r[THEMIS-DBG] INITIALIZE_PARTITION (seal)...");
        ioctl_noarg(self.fd.as_raw_fd(), THHV_INITIALIZE_PARTITION)
            .context("failed to initialize Themis partition")?;
        eprintln!("\r[THEMIS-DBG] INITIALIZE_PARTITION done");
        *initialized = true;

        // LAPIC timer emulation is set up per-vCPU in create_vcpu():
        // timerfd + irqfd + background thread.  WRMSR 0x6E0 exits
        // (trapped by the child MSR bitmap in capavisor) are handled
        // in handle_wrmsr_exit() which reprograms the timerfd.

        Ok(())
    }

    /// Unified policy-setting ioctl — wraps THHV_SET_POLICY.
    pub(super) fn set_policy(&self, kind: u64, key: u64, sub_key: u64, value: u64) -> anyhow::Result<()> {
        let mut sp = ThhvSetPolicy {
            kind,
            key,
            sub_key,
            value,
        };
        ioctl_with_mut_ref(self.fd.as_raw_fd(), THHV_SET_POLICY, &mut sp)
            .context("THHV_SET_POLICY failed")?;
        Ok(())
    }
}

/// Probe the Themis hypervisor CPUID feature leaf (0x4000_0001) to determine
/// whether hardware x2APIC virtualization (VIRT_X2APIC_MODE +
/// APIC_REGISTER_VIRT + VID) is available.  Capavisor sets
/// `FEATURE_X2APIC_VIRT` only when all three controls are in
/// `IA32_VMX_PROCBASED_CTLS2` allowed-1; nested KVM (L0) usually clears them.
///
/// Computed once and cached — CPUID exits trap to capavisor and we don't want
/// to add a vmexit on every policy push.
fn themis_x2apic_virt_supported() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHED: AtomicU8 = AtomicU8::new(0); // 0=unknown, 1=no, 2=yes
    match CACHED.load(Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    // SAFETY: CPUID is unprivileged. Capavisor intercepts the Themis leaf
    // range and synthesises the response.
    let supported = unsafe {
        let r = std::arch::x86_64::__cpuid(themis_abi::cpuid::LEAF_FEATURES);
        (r.eax & themis_abi::cpuid::feature_bits::FEATURE_X2APIC_VIRT) != 0
    };
    CACHED.store(if supported { 2 } else { 1 }, Ordering::Relaxed);
    eprintln!(
        "\r[THEMIS-DBG] Themis CPUID 0x40000001 -> FEATURE_X2APIC_VIRT={}",
        supported
    );
    supported
}
