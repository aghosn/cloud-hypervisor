//! Per-VM state shared between `ThemisVm` and its `ThemisVcpu`s.
//!
//! Holds the THHV partition fd, mmap'd meta pages, deferred-init queues
//! (memory regions), userspace I/O event matching, per-vCPU LAPIC state, and the
//! interposition-policy bookkeeping pushed before sealing the partition.

use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use vmm_sys_util::eventfd::EventFd;

use super::abi::*;
use super::consts::*;
use super::helpers::{ioctl_noarg, ioctl_with_mut_ref};
use super::mmap::MmapRegion;
use super::policy_walker::{
    PolicyOp, WalkContext, cpuid_policy_ranges, cpuid_range_segments, walk_api, walk_cpuid,
    walk_exits, walk_interrupts, walk_msrs,
};
use crate::arch::x86::CpuIdEntry;

pub(crate) struct ThemisVmState {
    pub fd: Arc<OwnedFd>,
    pub initialized: Mutex<bool>,
    pub initialization_failed: Mutex<bool>,
    pub pv: std::sync::OnceLock<super::pv::PvResources>,
    pub device_memory: crate::vm::SharedDeviceMemory,
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
    /// CPUID entries by VP index, pushed as scoped policy before sealing.
    pub cpuid_entries: Mutex<Vec<Vec<CpuIdEntry>>>,
    /// ivshmem BAR addresses for CPUID leaf 0x40000004 (Emulate override).
    /// Each entry: (index, bar0_gpa, bar2_gpa).
    pub ivshmem_bars: Mutex<Vec<(u32, u64, u64)>>,
    pub ioevents: Mutex<super::ioevent::IoEvents>,
    /// VTOM bit position (MAXPHYADDR - 1). Used to strip the VTOM bit from
    /// MMIO GPAs when the CoCo kernel marks device accesses as shared.
    pub vtom_bit: u32,
    /// Confidential mode: guest RAM is CARVEd (dom0 loses access) instead of
    /// ALIASed. MMIO regions remain ALIAS (shared by design).
    pub confidential: bool,
    /// Resolved policy config used to build the THHV_SET_POLICY stream at
    /// seal time.  Either the user's `--themis-config` file, or the bundled
    /// default profile (`standard.json` / `confidential.json`) selected on
    /// the `confidential` flag.
    pub themis_config: Arc<crate::themis::ThemisConfig>,
}

impl ThemisVmState {
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
        shared: &[super::config::SharedRegion],
    ) -> anyhow::Result<Vec<ThhvSetGuestMemory>> {
        let limit = 1u64
            .checked_shl(vtom_bit)
            .ok_or_else(|| anyhow::anyhow!("invalid vTOM bit"))?;
        for region in shared {
            anyhow::ensure!(
                region
                    .gpa
                    .checked_add(region.size)
                    .is_some_and(|end| end <= limit),
                "shared region exceeds private GPA address space",
            );
        }
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
            for shared in shared {
                let s_start = shared.gpa;
                let s_end = s_start + shared.size;
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

        Ok(out)
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

        // ── Config-driven baseline ─────────────────────────────────────
        //
        // MSR_DEFAULT + `msrs.overrides` come from the resolved ThemisConfig
        // (user file or bundled default profile).  The `policy_walker`
        // module produces the exact stream that used to be hardcoded here;
        // see `policy_walker::tests` for the byte-identical regression
        // check on the default profiles.
        let ctx = WalkContext {
            vtom_bit: self.vtom_bit,
        };
        let baseline_ops =
            walk_msrs(&self.themis_config, ctx).context("themis config: walk MSR policy")?;
        for op in &baseline_ops {
            self.emit_policy_op(op)?;
        }

        // ── Runtime-conditional: x2APIC virtualization ────────────────
        //
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
        // This block is NOT driven from the config yet because it depends
        // on runtime host-CPUID probing (`themis_x2apic_virt_supported`);
        // moving it into the config would let users override that probe,
        // which is out of scope for the initial apply refactor.  Kept as
        // an in-source runtime injection.
        if x2apic_virt {
            self.set_policy(policy_kind::MSR_RANGE, 0x800, 0x82F, 1)?; // Native
            self.set_policy(policy_kind::MSR_RANGE, 0x830, 0x830, 0)?; // Trap (ICR)
            if self.themis_config.general.uses_pv_devices() {
                self.set_policy(policy_kind::MSR_RANGE, 0x831, 0x83F, 1)?;
            } else {
                self.set_policy(policy_kind::MSR_RANGE, 0x831, 0x831, 1)?;
                self.set_policy(policy_kind::MSR_RANGE, 0x832, 0x832, 0)?; // Trap timer LVT
                self.set_policy(policy_kind::MSR_RANGE, 0x833, 0x83F, 1)?;
            }

            // IA32_APIC_BASE (0x1B): Emulate.  Stored value advertises the
            // architectural LAPIC base with EN (bit 11) and EXTD (bit 10) set
            // — Linux sees x2APIC already enabled and skips the xAPIC→x2APIC
            // transition entirely.  WRMSR is consumed (no-op) by capavisor's
            // msr_emulator handler to prevent the guest from disabling EXTD.
            //
            //   bit 8  (BSP):  supplied per-VP by capavisor's RDMSR emulator;
            //                  the remaining policy value is domain-wide.
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

        log::debug!(
            "Themis pushed MSR policy: {} baseline op(s), x2apic_virt={}",
            baseline_ops.len(),
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
        {
            let mut failed = self.initialization_failed.lock().unwrap();
            if *failed {
                anyhow::bail!("previous Themis initialization failed; recreate the partition");
            }
            *failed = true;
        }

        let api_op = walk_api(&self.themis_config).context("themis config: walk API policy")?;
        let exit_ops =
            walk_exits(&self.themis_config).context("themis config: walk VMEXIT policy")?;
        let interrupt_ops =
            walk_interrupts(&self.themis_config).context("themis config: walk interrupt policy")?;
        let ctx = WalkContext {
            vtom_bit: self.vtom_bit,
        };
        let (cpuid_emulate_ops, config_emulate_leaves) =
            walk_cpuid(&self.themis_config, ctx).context("themis config: walk CPUID overrides")?;

        // Flush all deferred SET_GUEST_MEMORY calls before sealing.
        // After this point dom0 loses EPT access to dom1's guest pages.
        let raw_regions: Vec<ThhvSetGuestMemory> =
            self.pending_memory.lock().unwrap().drain(..).collect();

        let bits = if self.confidential {
            self.vtom_bit
        } else {
            (core::arch::x86_64::__cpuid(0x80000008).eax & 0xFF).min(48)
        };
        let limit = 1u64
            .checked_shl(bits)
            .ok_or_else(|| anyhow::anyhow!("invalid GPA width"))?;
        let mut occupied = raw_regions
            .iter()
            .map(|r| {
                let gpa = r
                    .guest_pfn
                    .checked_mul(themis_abi::pv::PAGE_SIZE as u64)
                    .ok_or_else(|| anyhow::anyhow!("guest memory GPA overflow"))?;
                themis_abi::pv::descriptor::range_end(gpa, r.size, limit)
                    .map_err(|e| anyhow::anyhow!("invalid guest memory range: {e:?}"))?;
                Ok((gpa, r.size))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        occupied.push((
            super::consts::LAPIC_MMIO_BASE,
            themis_abi::pv::PAGE_SIZE as u64,
        ));

        // In confidential mode, split regions around known shared areas (EBDA/ACPI).
        // Shared regions become ALIAS slots (+ vTOM mirror), the rest stays CARVE.
        let mut regions = if self.confidential {
            Self::split_confidential_regions(
                &raw_regions,
                self.vtom_bit,
                self.themis_config.general.shared_regions(),
            )?
        } else {
            raw_regions
        };

        let num_vps = {
            let fds = self.vp_fds.lock().unwrap();
            if fds.iter().any(|&fd| fd < 0) {
                anyhow::bail!("PV launch requires contiguous created VP indices");
            }
            fds.len() as u32
        };
        if let Some(mut pv) =
            super::pv::PvResources::prepare(&self.themis_config, num_vps, &occupied, limit)?
        {
            let pv_start = u64::from(themis_abi::cpuid::LEAF_PV) << 32;
            let pv_end = pv_start | u64::from(u32::MAX);
            if cpuid_policy_ranges(&cpuid_emulate_ops)
                .iter()
                .any(|&(lo, hi)| lo <= pv_end && pv_start <= hi)
                || self
                    .cpuid_entries
                    .lock()
                    .unwrap()
                    .iter()
                    .flatten()
                    .any(|e| e.function == themis_abi::cpuid::LEAF_PV)
            {
                anyhow::bail!("PV discovery CPUID leaf conflicts with configured entries");
            }
            for entry in &self.themis_config.policies.cpuid.overrides {
                let conflicts = if entry.leaf == Some(themis_abi::cpuid::LEAF_PV) {
                    true
                } else if let Some(
                    [
                        super::config::CpuidCoord::Pair(start),
                        super::config::CpuidCoord::Pair(end),
                    ],
                ) = &entry.range
                {
                    let lo = super::policy_walker::resolve_scalar(&start[0], ctx)?;
                    let hi = super::policy_walker::resolve_scalar(&end[0], ctx)?;
                    (lo..=hi).contains(&u64::from(themis_abi::cpuid::LEAF_PV))
                } else {
                    false
                };
                if conflicts {
                    anyhow::bail!("PV discovery CPUID leaf overlaps an explicit policy override");
                }
            }
            pv.register_doorbells(self.fd.as_raw_fd(), limit)?;
            if self.themis_config.general.uses_pv_devices() {
                self.device_memory
                    .lock()
                    .unwrap()
                    .replace(pv.device_memory()?);
            }
            regions.extend(pv.memory_regions(if self.confidential {
                1u64 << self.vtom_bit
            } else {
                0
            }));
            self.pv
                .set(pv)
                .map_err(|_| anyhow::anyhow!("PV resources already initialized"))?;
        }

        log::debug!("Themis flushing {} memory regions", regions.len());
        for (i, mut region) in regions.iter().cloned().enumerate() {
            let kind = if region.flags & THHV_MEM_F_ALIAS != 0 {
                "ALIAS"
            } else {
                "CARVE"
            };
            let shmem_info = if region.shmem_mode != 0 {
                let path = std::str::from_utf8(&region.shmem_path)
                    .unwrap_or("<invalid>")
                    .trim_end_matches('\0');
                format!(
                    " shmem_mode={} count={} path=\"{}\"",
                    region.shmem_mode, region.shmem_count, path
                )
            } else {
                String::new()
            };
            log::debug!(
                "Themis SET_GUEST_MEMORY[{i}] gfn=0x{:x} size=0x{:x} {kind}{shmem_info}",
                region.guest_pfn,
                region.size
            );
            ioctl_with_mut_ref(self.fd.as_raw_fd(), THHV_SET_GUEST_MEMORY, &mut region)
                .context("deferred THHV_SET_GUEST_MEMORY failed")?;
            log::debug!("Themis SET_GUEST_MEMORY[{i}] done");
        }

        // Apply the config's VMEXIT routing before sealing. Previously the
        // JSON parser accepted `policies.exits`, but no THHV_SET_POLICY calls
        // were emitted, so every child silently retained the engine's
        // trap-all default.
        for op in &exit_ops {
            self.emit_policy_op(op)?;
        }
        log::debug!(
            "Themis pushed VMEXIT policy: {} op(s) from config",
            exit_ops.len()
        );

        // Apply the config's interrupt-routing policy before sealing.
        // Previously `policies.interrupts` was parsed/validated by config.rs
        // but never projected into THHV_SET_POLICY calls, so every child
        // domain's InterruptPolicy silently stayed at the engine's
        // `new_restricted()` default regardless of the JSON policy file.
        for op in &interrupt_ops {
            self.emit_policy_op(op)?;
        }
        log::debug!(
            "Themis pushed interrupt policy: {} op(s) from config",
            interrupt_ops.len()
        );

        // Push CPUID entries as Emulate overrides before sealing.
        // Each entry becomes 2 set_policy calls (word 0 and word 1).
        // Word 0: value = (eax << 32) | ebx — creates the entry.
        // Word 1: value = (ecx << 32) | edx — updates remaining fields.
        // Key encodes (leaf << 32 | subleaf), sub_key = word index.
        //
        // Patch leaf 1 ECX bit 21 (X2APIC) to 1 only when capavisor
        // advertises FEATURE_X2APIC_VIRT and we therefore pushed the
        // x2APIC MSR policy in ensure_msr_policy_pushed.  Forcing the bit
        // without that backing policy would let the guest switch to
        // x2APIC ops with no hardware virtualization, #GP'ing on the
        // first RDMSR 0x802.
        let x2apic_virt = themis_x2apic_virt_supported();
        let mut cpuid_entries = self.cpuid_entries.lock().unwrap().clone();
        if x2apic_virt {
            for entry in cpuid_entries.iter_mut().flatten() {
                if entry.function == 1 && entry.index == 0 {
                    entry.ecx |= 1 << 21; // X2APIC
                }
            }
        }

        // Push ivshmem BAR addresses as Emulate CPUID leaf 0x40000004.
        // Each ivshmem device gets its own subleaf (index).
        // EAX = BAR0 GPA (32-bit), EBX = device count,
        // ECX = BAR2 GPA (low 32), EDX = BAR2 GPA (high 32).
        let ivshmem_bars = self.ivshmem_bars.lock().unwrap().clone();
        if !ivshmem_bars.is_empty() {
            let count = ivshmem_bars.len() as u32;
            log::debug!("Themis pushing {count} ivshmem CPUID entries (leaf 0x40000004)");
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
                log::debug!("Themis ivshmem[{index}]: bar0=0x{bar0_gpa:x} bar2=0x{bar2_gpa:x}");
            }
        }

        // Push Native ranges for Themis hypervisor CPUID leaves so the child
        // can discover the capavisor.  The capavisor handles these natively
        // (signature, features, domcomm, limits, etc.).
        //
        // The engine rejects overlapping entries. Exclude runtime discovery
        // leaves and every configured override from the automatic Native range.

        let range_start: u32 = 0x40000000;
        let range_end: u32 = 0x4FFFFFFF;
        let mut emulate_leaves: Vec<u32> = Vec::new();
        if let Some(pv) = self.pv.get() {
            let key = u64::from(themis_abi::cpuid::LEAF_PV) << 32;
            let word0 = ((pv.descriptor_gpa as u32 as u64) << 32) | (pv.descriptor_gpa >> 32);
            let word1 = ((themis_abi::pv::PAGE_SIZE as u64) << 32)
                | u64::from(themis_abi::pv::descriptor::VERSION);
            self.set_policy(policy_kind::CPUID_EMULATE, key, 0, word0)?;
            self.set_policy(policy_kind::CPUID_EMULATE, key, 1, word1)?;
            let identity = key | u64::from(themis_abi::cpuid::PV_SUBLEAF_IDENTITY);
            self.set_policy(policy_kind::CPUID_RANGE, identity, identity, 1)?;
            emulate_leaves.push(themis_abi::cpuid::LEAF_PV);
        }
        if !ivshmem_bars.is_empty() {
            emulate_leaves.push(0x40000004);
        }
        emulate_leaves.extend(config_emulate_leaves.iter().copied());

        if self.themis_config.policies.cpuid.hypervisor_range_native {
            let mut excluded = cpuid_policy_ranges(&cpuid_emulate_ops);
            excluded.extend(emulate_leaves.iter().map(|&leaf| {
                let start = u64::from(leaf) << 32;
                (start, start | u64::from(u32::MAX))
            }));
            for (lo, hi) in cpuid_range_segments(
                u64::from(range_start) << 32,
                (u64::from(range_end) << 32) | u64::from(u32::MAX),
                &excluded,
            ) {
                self.set_policy(policy_kind::CPUID_RANGE, lo, hi, 1)?;
            }
            log::debug!(
                "Themis pushed Native CPUID ranges (split around {} emulate leaves)",
                emulate_leaves.len()
            );
        }

        // Emit config-declared CPUID Emulate overrides (e.g. CoCo leaf
        // 0x40000100 with the vTOM signature).  Order matches
        // vm_state.rs pre-refactor: after Native ranges, before seal.
        for op in &cpuid_emulate_ops {
            self.emit_policy_op(op)?;
        }
        if !cpuid_emulate_ops.is_empty() {
            log::debug!(
                "Themis pushed {} CPUID policy op(s) from config",
                cpuid_emulate_ops.len()
            );
        }

        // Install common discovery/configuration first: each explicit VP
        // CPUID category starts from that complete domain baseline.
        for (vp, entries) in cpuid_entries.iter().enumerate() {
            log::debug!(
                "Themis VP {vp} pre-seal CPUID table: {} entries ({} flagged)",
                entries.len(),
                entries.iter().filter(|entry| entry.flags != 0).count()
            );
        }
        for op in super::policy_walker::walk_vp_cpuid(&cpuid_entries, &cpuid_emulate_ops)? {
            self.emit_policy_op(&op)?;
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

        self.emit_policy_op(&api_op)?;
        log::debug!("Themis INITIALIZE_PARTITION (seal)...");
        ioctl_noarg(self.fd.as_raw_fd(), THHV_INITIALIZE_PARTITION)
            .context("failed to initialize Themis partition")?;
        log::debug!("Themis INITIALIZE_PARTITION done");
        *initialized = true;
        *self.initialization_failed.lock().unwrap() = false;

        // LAPIC timer emulation is set up per-vCPU in create_vcpu():
        // timerfd + irqfd + background thread.  WRMSR 0x6E0 exits
        // (trapped by the child MSR bitmap in capavisor) are handled
        // in handle_wrmsr_exit() which reprograms the timerfd.

        Ok(())
    }

    /// Unified policy-setting ioctl — wraps THHV_SET_POLICY.
    pub(super) fn set_policy(
        &self,
        kind: u64,
        key: u64,
        sub_key: u64,
        value: u64,
    ) -> anyhow::Result<()> {
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

    /// Push a walker-emitted [`PolicyOp`].  Thin adapter around
    /// [`Self::set_policy`] — kept as its own method so the walker doesn't
    /// have to know about ThemisVmState fields.
    pub(super) fn emit_policy_op(&self, op: &PolicyOp) -> anyhow::Result<()> {
        self.set_policy(op.kind, op.key, op.sub_key, op.value)
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
    let r = std::arch::x86_64::__cpuid(themis_abi::cpuid::LEAF_FEATURES);
    let supported = (r.eax & themis_abi::cpuid::feature_bits::FEATURE_X2APIC_VIRT) != 0;
    CACHED.store(if supported { 2 } else { 1 }, Ordering::Relaxed);
    log::debug!("Themis CPUID 0x40000001 -> FEATURE_X2APIC_VIRT={supported}");
    supported
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidential_split_uses_declared_ranges_and_honors_explicit_empty() {
        let cfg = super::super::config::ThemisConfig::from_json_str(
            r#"{"general":{"confidential":true,"shared_regions":[{"gpa":"0x3000","size":"0x2000"}]}}"#,
        ).unwrap();
        let input = [ThhvSetGuestMemory {
            userspace_addr: 0x1000_0000,
            size: 0x10_0000,
            ..Default::default()
        }];
        let regions =
            ThemisVmState::split_confidential_regions(&input, 39, cfg.general.shared_regions())
                .unwrap();
        let shared = regions
            .iter()
            .filter(|r| r.flags & THHV_MEM_F_ALIAS != 0)
            .collect::<Vec<_>>();
        assert_eq!(shared.len(), 2);
        assert_eq!(
            (shared[0].guest_pfn << 12, shared[0].size),
            (0x3000, 0x2000)
        );
        assert_eq!(shared[1].guest_pfn << 12, (1 << 39) | 0x3000);
        assert_eq!(shared[0].userspace_addr, 0x1000_3000);
        assert!(
            regions
                .iter()
                .any(|r| r.guest_pfn << 12 == 0x5000 && r.flags == 0)
        );
        let private = ThemisVmState::split_confidential_regions(&input, 39, &[]).unwrap();
        assert_eq!(private.len(), 1);
        assert_eq!(private[0].flags, 0);
        ThemisVmState::split_confidential_regions(&input, 12, cfg.general.shared_regions())
            .unwrap_err();
    }

    #[test]
    fn confidential_ap_firmware_has_no_parent_alias_or_vtom_mirror() {
        let host_base = 0x1000_0000;
        let regions = ThemisVmState::split_confidential_regions(
            &[ThhvSetGuestMemory {
                userspace_addr: host_base,
                size: 0x20_0000,
                ..ThhvSetGuestMemory::default()
            }],
            39,
            super::super::config::GeneralConfig::default().shared_regions(),
        )
        .unwrap();
        let firmware = regions
            .iter()
            .find(|region| region.guest_pfn << 12 == 0xe8000)
            .unwrap();
        assert_eq!(firmware.size, 0x5000);
        assert_eq!(firmware.flags, 0);
        assert_eq!(firmware.userspace_addr, host_base + 0xe8000);

        for region in &regions {
            let start = region.guest_pfn << 12;
            let end = start + region.size;
            let mirror = 1u64 << 39;
            assert!(end <= mirror + 0xe8000 || start >= mirror + 0xed000);
            if region.flags & THHV_MEM_F_ALIAS != 0 {
                assert!(end <= 0xe8000 || start >= 0xed000);
            }
        }
        for gpa in [0xa0000, 0xed000, (1 << 39) | 0xa0000, (1 << 39) | 0xed000] {
            assert!(regions.iter().any(|region| {
                region.guest_pfn << 12 == gpa && region.flags & THHV_MEM_F_ALIAS != 0
            }));
        }
    }
}
