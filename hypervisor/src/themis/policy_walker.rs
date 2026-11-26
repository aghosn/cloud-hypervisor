//! Walks a [`ThemisConfig`] and emits `THHV_SET_POLICY` tuples in the
//! order the seal path (`vm_state.rs::ensure_msr_policy_pushed` +
//! `ensure_initialized`) expects.
//!
//! Design constraint (see `docs/chv-themis-config.md` v0.4 and the
//! `chv-themis-config-apply` todo): when a caller loads the bundled
//! default profile matching its `confidential` flag, the emitted
//! MSR and CPUID Emulate operations preserve the previously-hardcoded stream
//! in `vm_state.rs`. CPUID's declared default is also emitted explicitly;
//! it must not silently remain the engine's restricted-child Trap default.
//!
//! Defaults and explicit overrides drive API, interrupt, exit, MSR and
//! CPUID policy. Per-VP CPUID tables are split around configured overrides.
//!
//! The runtime-dependent injections (x2APIC MSR block, per-VM CPUID
//! entries, ivshmem BAR CPUID leaves, ranges split around emulate
//! leaves) are still layered on top by `vm_state.rs` — this walker
//! provides the helpers they need to compose the final stream.

use themis_abi::regs::VpRegister;
use themis_abi::vmx_exit_reasons;

use super::config::{
    ApiAllow, CpuidCoord, DefaultAction, ExitReason, MsrOverride, OverrideAction, RegSet,
    RegSetSymbolic, ThemisConfig, ValueOrExpr, Visibility,
};
use super::consts::policy_kind;
use crate::arch::x86::{CPUID_FLAG_VALID_INDEX, CpuIdEntry};

/// A single `THHV_SET_POLICY` operation, in engine wire form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PolicyOp {
    pub kind: u64,
    pub key: u64,
    pub sub_key: u64,
    pub value: u64,
}

impl PolicyOp {
    pub(super) fn new(kind: u64, key: u64, sub_key: u64, value: u64) -> Self {
        Self {
            kind,
            key,
            sub_key,
            value,
        }
    }
}

/// Runtime context needed to resolve config-time sentinels (e.g.
/// `"auto:vtom_bit"`).
#[derive(Clone, Copy, Debug)]
pub(super) struct WalkContext {
    pub vtom_bit: u32,
}

/// Errors raised while resolving a config value into a concrete
/// 64-bit-ish number.  Bubbled up to the caller as `anyhow::Error`.
#[derive(Debug, thiserror::Error)]
pub(super) enum WalkError {
    #[error("unknown emulate value expression: {0:?}")]
    UnknownExpression(String),
    #[error("emulate override at msr {msr:#x} is missing its stored value")]
    MissingMsrValue { msr: u32 },
    #[error("cpuid override at leaf {leaf:#x} sub {sub:#x} is missing its register values")]
    MissingCpuidValues { leaf: u32, sub: u32 },
    #[error(
        "cpuid override uses a range with action Emulate (unsupported: emulate needs per-leaf values)"
    )]
    EmulateOverRange,
    #[error("msr override has neither `msr` nor `range` set")]
    MsrOverrideMissingTarget,
    #[error("msr range [{lo:#x},{hi:#x}] with action Emulate — Emulate requires a single MSR")]
    EmulateOverMsrRange { lo: u32, hi: u32 },
    #[error("unknown VMEXIT reason: {0:?}")]
    UnknownExitReason(String),
    #[error("unknown VP register name in register set: {0:?}")]
    UnknownRegister(String),
    #[error("PV exit {0:#x} must trap with empty parent register masks")]
    UnsafePvExit(u32),
    #[error("invalid CPUID VP index: {0}")]
    InvalidVpIndex(usize),
    #[error("ambiguous or unsupported CPUID entries for VP {vp}, leaf {leaf:#x}")]
    InvalidCpuidTable { vp: usize, leaf: u32 },
    #[error("unknown monitor API permission: {0:?}")]
    UnknownMonitorApi(String),
    #[error("invalid, overlapping, or out-of-range CPUID override")]
    InvalidCpuidOverride,
}

pub(super) fn walk_api(cfg: &ThemisConfig) -> Result<PolicyOp, WalkError> {
    let bits = match &cfg.policies.api.allow {
        ApiAllow::All => 0xffff,
        ApiAllow::List(names) => {
            let mut bits = 0;
            for name in names {
                let bit = match name.to_ascii_uppercase().as_str() {
                    "CREATE" => 0,
                    "SET" => 1,
                    "GET" => 2,
                    "SEND" => 3,
                    "SEAL" => 4,
                    "ATTEST" => 5,
                    "ENUMERATE" => 6,
                    "SWITCH" => 7,
                    "ALIAS" => 8,
                    "CARVE" => 9,
                    "REVOKE" => 10,
                    "GETCHAN" | "GET_CHAN" => 11,
                    "RECEIVE_AFTER_SEAL" => 12,
                    "MAP_SELF" => 13,
                    "CREATE_DOORBELL" => 14,
                    "NOTIFY" => 15,
                    _ => return Err(WalkError::UnknownMonitorApi(name.clone())),
                };
                bits |= 1u64 << bit;
            }
            bits
        }
    };
    Ok(PolicyOp::new(policy_kind::API_MONITOR, 0, 0, bits))
}

// ── Interrupt policy walker ──────────────────────────────────────────────── //

/// Map a config `Visibility` to the engine's `InterruptVisibility` wire value.
/// Ordering must match `capa-engine`'s `visibility_to_u64`: Deliver=0,
/// Report=1, NotReport=2.
fn visibility_wire_value(vis: Visibility) -> u64 {
    match vis {
        Visibility::Deliver => 0,
        Visibility::Report => 1,
        Visibility::NotReport => 2,
    }
}

/// Emit the interrupt-routing policy stream for `cfg`:
///   1. `DEFAULT_INTR_VISIBILITY`
///   2. Per-vector overrides: visibility, read_set (per word), write_set
///      (per word), and the `injectable` flag.
///
/// Defaults are emitted before overrides so a new override starts from
/// the configured default rather than an implicit engine register mask.
pub(super) fn walk_interrupts(cfg: &ThemisConfig) -> Result<Vec<PolicyOp>, WalkError> {
    let mut out = vec![PolicyOp::new(
        policy_kind::DEFAULT_INTR_VISIBILITY,
        0,
        0,
        visibility_wire_value(cfg.policies.interrupts.default.visibility),
    )];
    for (kind, set) in [
        (
            policy_kind::DEFAULT_INTR_REG_READ_SET,
            &cfg.policies.interrupts.default.read_set,
        ),
        (
            policy_kind::DEFAULT_INTR_REG_WRITE_SET,
            &cfg.policies.interrupts.default.write_set,
        ),
    ] {
        for (word, bits) in reg_set_words(set)?.into_iter().enumerate() {
            out.push(PolicyOp::new(kind, 0, word as u64, bits));
        }
    }
    out.push(PolicyOp::new(
        policy_kind::DEFAULT_VECTOR_INJECTABLE,
        0,
        0,
        u64::from(cfg.policies.interrupts.default.injectable),
    ));

    for ov in &cfg.policies.interrupts.overrides {
        let vector = ov.vector as u64;
        out.push(PolicyOp::new(
            policy_kind::VECTOR_VISIBILITY,
            vector,
            0,
            visibility_wire_value(ov.policy.visibility),
        ));

        for (word, bits) in reg_set_words(&ov.policy.read_set)?.into_iter().enumerate() {
            out.push(PolicyOp::new(
                policy_kind::VECTOR_REG_READ_SET,
                vector,
                word as u64,
                bits,
            ));
        }
        for (word, bits) in reg_set_words(&ov.policy.write_set)?.into_iter().enumerate() {
            out.push(PolicyOp::new(
                policy_kind::VECTOR_REG_WRITE_SET,
                vector,
                word as u64,
                bits,
            ));
        }

        out.push(PolicyOp::new(
            policy_kind::VECTOR_INJECTABLE,
            vector,
            0,
            u64::from(ov.policy.injectable),
        ));
    }

    Ok(out)
}

// ── VMEXIT walker ───────────────────────────────────────────────────────── //

/// Emit the VMEXIT policy stream for `cfg`.
///
/// Emits the complete default action before per-reason overrides.
pub(super) fn walk_exits(cfg: &ThemisConfig) -> Result<Vec<PolicyOp>, WalkError> {
    let mut out = vec![PolicyOp::new(
        policy_kind::DEFAULT_EXIT_TRAP,
        0,
        0,
        u64::from(cfg.policies.exits.default.trap),
    )];
    for (kind, set) in [
        (
            policy_kind::DEFAULT_EXIT_REG_READ_SET,
            &cfg.policies.exits.default.read_set,
        ),
        (
            policy_kind::DEFAULT_EXIT_REG_WRITE_SET,
            &cfg.policies.exits.default.write_set,
        ),
    ] {
        for (word, bits) in reg_set_words(set)?.into_iter().enumerate() {
            out.push(PolicyOp::new(kind, 0, word as u64, bits));
        }
    }

    for ov in &cfg.policies.exits.overrides {
        let reason = resolve_exit_reason(&ov.reason)?;
        if has_pv_channels(cfg)
            && pv_exit(reason)
            && (!ov.action.trap
                || reg_set_words(&ov.action.read_set)? != [0; 3]
                || reg_set_words(&ov.action.write_set)? != [0; 3])
        {
            return Err(WalkError::UnsafePvExit(reason));
        }
        out.push(PolicyOp::new(
            policy_kind::EXIT_REASON_TRAP,
            reason as u64,
            0,
            u64::from(ov.action.trap),
        ));

        for (word, bits) in reg_set_words(&ov.action.read_set)?.into_iter().enumerate() {
            out.push(PolicyOp::new(
                policy_kind::EXIT_REASON_REG_READ_SET,
                reason as u64,
                word as u64,
                bits,
            ));
        }
        for (word, bits) in reg_set_words(&ov.action.write_set)?.into_iter().enumerate() {
            out.push(PolicyOp::new(
                policy_kind::EXIT_REASON_REG_WRITE_SET,
                reason as u64,
                word as u64,
                bits,
            ));
        }
    }

    if has_pv_channels(cfg) {
        for reason in [themis_abi::synthetic_exits::THEMIS_EXIT_YIELD] {
            out.push(PolicyOp::new(
                policy_kind::EXIT_REASON_TRAP,
                u64::from(reason),
                0,
                1,
            ));
            for kind in [
                policy_kind::EXIT_REASON_REG_READ_SET,
                policy_kind::EXIT_REASON_REG_WRITE_SET,
            ] {
                for word in 0..3 {
                    out.push(PolicyOp::new(kind, u64::from(reason), word, 0));
                }
            }
        }
    }
    Ok(out)
}

pub(super) fn walk_vp_cpuid(
    tables: &[Vec<CpuIdEntry>],
    configured: &[PolicyOp],
) -> Result<Vec<PolicyOp>, WalkError> {
    let mut out = Vec::new();
    for (vp, entries) in tables.iter().enumerate() {
        let kind = u32::try_from(vp)
            .ok()
            .and_then(|vp| policy_kind::for_vp(policy_kind::CPUID_EMULATE_RANGE, vp))
            .ok_or(WalkError::InvalidVpIndex(vp))?;
        let mut leaves = std::collections::BTreeMap::<u32, Vec<&CpuIdEntry>>::new();
        for entry in entries {
            leaves.entry(entry.function).or_default().push(entry);
        }
        for (leaf, mut entries) in leaves {
            let indexed = entries[0].flags & CPUID_FLAG_VALID_INDEX != 0;
            entries.sort_by_key(|entry| entry.index);
            if entries.iter().any(|entry| {
                entry.flags & !CPUID_FLAG_VALID_INDEX != 0
                    || (entry.flags & CPUID_FLAG_VALID_INDEX != 0) != indexed
            }) || (!indexed && entries.len() != 1)
                || entries
                    .windows(2)
                    .any(|pair| pair[0].index == pair[1].index)
            {
                return Err(WalkError::InvalidCpuidTable { vp, leaf });
            }
            let configured_ranges = cpuid_policy_ranges(configured);
            if !indexed {
                push_cpuid_range(
                    &mut out,
                    kind,
                    leaf,
                    (0, u32::MAX),
                    entries[0],
                    &configured_ranges,
                );
                continue;
            }
            let zero = CpuIdEntry::default();
            let mut next = 0u64;
            for entry in entries {
                if next < u64::from(entry.index) {
                    push_cpuid_range(
                        &mut out,
                        kind,
                        leaf,
                        (next as u32, entry.index - 1),
                        &zero,
                        &configured_ranges,
                    );
                }
                push_cpuid_range(
                    &mut out,
                    kind,
                    leaf,
                    (entry.index, entry.index),
                    entry,
                    &configured_ranges,
                );
                next = u64::from(entry.index) + 1;
            }
            if next <= u64::from(u32::MAX) {
                push_cpuid_range(
                    &mut out,
                    kind,
                    leaf,
                    (next as u32, u32::MAX),
                    &zero,
                    &configured_ranges,
                );
            }
        }
        if !entries.is_empty() {
            for op in configured {
                let kind =
                    policy_kind::for_vp(op.kind, vp as u32).ok_or(WalkError::InvalidVpIndex(vp))?;
                out.push(PolicyOp { kind, ..*op });
            }
        }
    }
    Ok(out)
}

fn push_cpuid_range(
    out: &mut Vec<PolicyOp>,
    kind: u64,
    leaf: u32,
    (start, end): (u32, u32),
    entry: &CpuIdEntry,
    excluded: &[(u64, u64)],
) {
    let words = [
        (u64::from(entry.eax) << 32) | u64::from(entry.ebx),
        (u64::from(entry.ecx) << 32) | u64::from(entry.edx),
    ];
    let mut emit = |lo: u32, hi: u32| {
        let key = (u64::from(leaf) << 32) | u64::from(lo);
        let end = (u64::from(leaf) << 32) | u64::from(hi);
        for (word, value) in words.into_iter().enumerate() {
            out.push(PolicyOp::new(
                policy_kind::with_emulate_word(kind, word as u8)
                    .expect("validated CPUID range selector"),
                key,
                end,
                value,
            ));
        }
    };
    for (lo, hi) in cpuid_range_segments(
        (u64::from(leaf) << 32) | u64::from(start),
        (u64::from(leaf) << 32) | u64::from(end),
        excluded,
    ) {
        emit(lo as u32, hi as u32);
    }
}

pub(super) fn cpuid_policy_ranges(ops: &[PolicyOp]) -> Vec<(u64, u64)> {
    ops.iter()
        .filter_map(|op| match op.kind {
            policy_kind::CPUID_RANGE => Some((op.key, op.sub_key)),
            policy_kind::CPUID_EMULATE if op.sub_key == 0 => Some((op.key, op.key)),
            _ => None,
        })
        .collect()
}

pub(super) fn cpuid_range_segments(
    start: u64,
    end: u64,
    excluded: &[(u64, u64)],
) -> Vec<(u64, u64)> {
    let mut excluded = excluded.to_vec();
    excluded.sort_unstable();
    let mut out = Vec::new();
    let mut next = start;
    for (lo, hi) in excluded {
        if hi < next || lo > end {
            continue;
        }
        if next < lo {
            out.push((next, lo - 1));
        }
        if hi >= end {
            return out;
        }
        next = hi + 1;
    }
    if next <= end {
        out.push((next, end));
    }
    out
}

fn has_pv_channels(cfg: &ThemisConfig) -> bool {
    cfg.comm
        .as_ref()
        .is_some_and(|comm| !comm.channels.is_empty())
}

fn pv_exit(reason: u32) -> bool {
    reason == themis_abi::synthetic_exits::THEMIS_EXIT_YIELD
}

fn resolve_exit_reason(reason: &ExitReason) -> Result<u32, WalkError> {
    let ExitReason::Name(name) = reason else {
        return Ok(match reason {
            ExitReason::Number(value) => *value,
            ExitReason::Name(_) => unreachable!(),
        });
    };

    if let Some(hex) = name.strip_prefix("0x").or_else(|| name.strip_prefix("0X")) {
        return u32::from_str_radix(hex, 16)
            .map_err(|_| WalkError::UnknownExitReason(name.clone()));
    }
    if let Ok(value) = name.parse::<u32>() {
        return Ok(value);
    }

    match name.as_str() {
        "THEMIS_EXIT_YIELD" => Ok(themis_abi::synthetic_exits::THEMIS_EXIT_YIELD),
        "EXCEPTION_NMI" => Ok(vmx_exit_reasons::EXCEPTION_NMI),
        "EXTERNAL_INTERRUPT" => Ok(vmx_exit_reasons::EXTERNAL_INTERRUPT),
        "TRIPLE_FAULT" => Ok(vmx_exit_reasons::TRIPLE_FAULT),
        "INIT_SIGNAL" => Ok(vmx_exit_reasons::INIT_SIGNAL),
        "SIPI" => Ok(vmx_exit_reasons::SIPI),
        "INTERRUPT_WINDOW" => Ok(vmx_exit_reasons::INTERRUPT_WINDOW),
        "CPUID" => Ok(vmx_exit_reasons::CPUID),
        "HLT" => Ok(vmx_exit_reasons::HLT),
        "VMCALL" => Ok(vmx_exit_reasons::VMCALL),
        "CR_ACCESS" => Ok(vmx_exit_reasons::CR_ACCESS),
        "IO_INSTRUCTION" => Ok(vmx_exit_reasons::IO_INSTRUCTION),
        "RDMSR" => Ok(vmx_exit_reasons::RDMSR),
        "WRMSR" => Ok(vmx_exit_reasons::WRMSR),
        "VMENTRY_INVALID_GUEST" => Ok(vmx_exit_reasons::VMENTRY_INVALID_GUEST),
        "APIC_ACCESS" => Ok(vmx_exit_reasons::APIC_ACCESS),
        "EOI_INDUCED" => Ok(vmx_exit_reasons::EOI_INDUCED),
        "EPT_VIOLATION" => Ok(vmx_exit_reasons::EPT_VIOLATION),
        "EPT_MISCONFIG" => Ok(vmx_exit_reasons::EPT_MISCONFIG),
        "VMX_PREEMPTION_TIMER" => Ok(vmx_exit_reasons::VMX_PREEMPTION_TIMER),
        "XSETBV" => Ok(vmx_exit_reasons::XSETBV),
        "APIC_WRITE" => Ok(vmx_exit_reasons::APIC_WRITE),
        _ => Err(WalkError::UnknownExitReason(name.clone())),
    }
}

fn reg_set_words(set: &RegSet) -> Result<[u64; 3], WalkError> {
    match set {
        RegSet::Symbolic(RegSetSymbolic::All) => Ok([u64::MAX; 3]),
        RegSet::Symbolic(RegSetSymbolic::None) => Ok([0; 3]),
        RegSet::Bitmap(bits) => Ok([*bits, 0, 0]),
        RegSet::List(names) => {
            let mut words = [0; 3];
            for name in names {
                let id = register_id(name)?;
                words[(id / 64) as usize] |= 1u64 << (id % 64);
            }
            Ok(words)
        }
    }
}

fn register_id(name: &str) -> Result<u64, WalkError> {
    let reg = match name {
        "RAX" => VpRegister::Rax,
        "RBX" => VpRegister::Rbx,
        "RCX" => VpRegister::Rcx,
        "RDX" => VpRegister::Rdx,
        "RSI" => VpRegister::Rsi,
        "RDI" => VpRegister::Rdi,
        "RBP" => VpRegister::Rbp,
        "R8" => VpRegister::R8,
        "R9" => VpRegister::R9,
        "R10" => VpRegister::R10,
        "R11" => VpRegister::R11,
        "R12" => VpRegister::R12,
        "R13" => VpRegister::R13,
        "R14" => VpRegister::R14,
        "R15" => VpRegister::R15,
        "RSP" => VpRegister::Rsp,
        "RIP" => VpRegister::Rip,
        "RFLAGS" => VpRegister::Rflags,
        "CR0" => VpRegister::Cr0,
        "CR3" => VpRegister::Cr3,
        "CR4" => VpRegister::Cr4,
        "EFER" => VpRegister::Efer,
        "DR7" => VpRegister::Dr7,
        _ => return Err(WalkError::UnknownRegister(name.to_string())),
    };
    Ok(reg as u64)
}

// ── MSR walker ─────────────────────────────────────────────────────────── //

/// Emit the MSR policy stream for `cfg` (in `vm_state.rs`-native order):
///   1. `MSR_DEFAULT`
///   2. Each `MsrOverride` in declaration order
pub(super) fn walk_msrs(cfg: &ThemisConfig, ctx: WalkContext) -> Result<Vec<PolicyOp>, WalkError> {
    let mut out = Vec::new();

    // MSR_DEFAULT encoding: value = 0 (Trap) or 1 (Native).
    out.push(PolicyOp::new(
        policy_kind::MSR_DEFAULT,
        0,
        0,
        default_action_value(cfg.policies.msrs.default),
    ));

    for ov in &cfg.policies.msrs.overrides {
        push_msr_override(&mut out, ov, ctx)?;
    }
    Ok(out)
}

fn push_msr_override(
    out: &mut Vec<PolicyOp>,
    ov: &MsrOverride,
    ctx: WalkContext,
) -> Result<(), WalkError> {
    match ov.action {
        OverrideAction::Emulate => {
            let msr = match (ov.msr, ov.range) {
                (Some(m), None) => m,
                (None, Some([lo, hi])) => return Err(WalkError::EmulateOverMsrRange { lo, hi }),
                _ => return Err(WalkError::MsrOverrideMissingTarget),
            };
            let value = ov
                .value
                .as_ref()
                .ok_or(WalkError::MissingMsrValue { msr })
                .and_then(|v| resolve_scalar(v, ctx))?;

            // vm_state.rs emits sub_key=0 with the full u64 today (see the
            // 0x6E0 push at line ~237).  For byte-identical output, if the
            // stored value fits in u32 we emit only sub_key=0 (matches
            // 0x6E0=0).  If it needs the high 32 bits, emit both — matches
            // the runtime IA32_APIC_BASE injection at line ~281.
            if (value >> 32) == 0 {
                out.push(PolicyOp::new(
                    policy_kind::MSR_EMULATE,
                    msr as u64,
                    0,
                    value,
                ));
            } else {
                out.push(PolicyOp::new(
                    policy_kind::MSR_EMULATE,
                    msr as u64,
                    0,
                    value & 0xFFFF_FFFF,
                ));
                out.push(PolicyOp::new(
                    policy_kind::MSR_EMULATE,
                    msr as u64,
                    1,
                    value >> 32,
                ));
            }
        }
        OverrideAction::Native | OverrideAction::Trap => {
            let action_val = match ov.action {
                OverrideAction::Native => 1,
                OverrideAction::Trap => 0,
                OverrideAction::Emulate => unreachable!(),
            };
            match (ov.msr, ov.range) {
                (Some(m), None) => out.push(PolicyOp::new(
                    policy_kind::MSR_RANGE,
                    m as u64,
                    m as u64,
                    action_val,
                )),
                (None, Some([lo, hi])) => out.push(PolicyOp::new(
                    policy_kind::MSR_RANGE,
                    lo as u64,
                    hi as u64,
                    action_val,
                )),
                _ => return Err(WalkError::MsrOverrideMissingTarget),
            }
        }
    }
    Ok(())
}

// ── CPUID walker ───────────────────────────────────────────────────────── //

/// Emit the CPUID default and all explicit overrides.
///
/// Returns the emitted ops plus the set of `(leaf)` numbers that should
/// be excluded from any Native hypervisor-range push.
pub(super) fn walk_cpuid(
    cfg: &ThemisConfig,
    ctx: WalkContext,
) -> Result<(Vec<PolicyOp>, Vec<u32>), WalkError> {
    let mut out = vec![PolicyOp::new(
        policy_kind::CPUID_DEFAULT,
        0,
        0,
        default_action_value(cfg.policies.cpuid.default),
    )];
    let mut emulate_leaves = Vec::new();
    let mut ranges = Vec::new();

    for ov in &cfg.policies.cpuid.overrides {
        let coordinate = |coord: &CpuidCoord| -> Result<u64, WalkError> {
            let CpuidCoord::Pair([leaf, subleaf]) = coord;
            let leaf = u32::try_from(resolve_scalar(leaf, ctx)?)
                .map_err(|_| WalkError::InvalidCpuidOverride)?;
            let subleaf = u32::try_from(resolve_scalar(subleaf, ctx)?)
                .map_err(|_| WalkError::InvalidCpuidOverride)?;
            Ok((u64::from(leaf) << 32) | u64::from(subleaf))
        };
        let (start, end) = match (ov.leaf, ov.subleaf, &ov.range) {
            (Some(leaf), subleaf, None) => {
                let key = (u64::from(leaf) << 32) | u64::from(subleaf.unwrap_or(0));
                (key, key)
            }
            (None, None, Some([lo, hi])) => (coordinate(lo)?, coordinate(hi)?),
            _ => return Err(WalkError::InvalidCpuidOverride),
        };
        if end < start || ranges.iter().any(|&(lo, hi)| start <= hi && lo <= end) {
            return Err(WalkError::InvalidCpuidOverride);
        }
        ranges.push((start, end));
        if matches!(ov.action, OverrideAction::Emulate) {
            let (leaf, subleaf) = match (ov.leaf, ov.subleaf, &ov.range) {
                (Some(l), s, None) => (l, s.unwrap_or(0)),
                (None, _, Some(_)) => return Err(WalkError::EmulateOverRange),
                _ => return Err(WalkError::MissingCpuidValues { leaf: 0, sub: 0 }),
            };

            let eax = ov
                .eax
                .as_ref()
                .map(|v| resolve_scalar(v, ctx))
                .transpose()?;
            let ebx = ov
                .ebx
                .as_ref()
                .map(|v| resolve_scalar(v, ctx))
                .transpose()?;
            let ecx = ov
                .ecx
                .as_ref()
                .map(|v| resolve_scalar(v, ctx))
                .transpose()?;
            let edx = ov
                .edx
                .as_ref()
                .map(|v| resolve_scalar(v, ctx))
                .transpose()?;

            if eax.is_none() && ebx.is_none() && ecx.is_none() && edx.is_none() {
                return Err(WalkError::MissingCpuidValues { leaf, sub: subleaf });
            }

            let register = |value: Option<u64>| {
                u32::try_from(value.unwrap_or(0)).map_err(|_| WalkError::InvalidCpuidOverride)
            };
            let (eax, ebx, ecx, edx) = (
                register(eax)?,
                register(ebx)?,
                register(ecx)?,
                register(edx)?,
            );

            let key = ((leaf as u64) << 32) | (subleaf as u64);
            let word0 = ((eax as u64) << 32) | (ebx as u64);
            let word1 = ((ecx as u64) << 32) | (edx as u64);
            out.push(PolicyOp::new(policy_kind::CPUID_EMULATE, key, 0, word0));
            out.push(PolicyOp::new(policy_kind::CPUID_EMULATE, key, 1, word1));
            emulate_leaves.push(leaf);
        } else {
            if ov.eax.is_some() || ov.ebx.is_some() || ov.ecx.is_some() || ov.edx.is_some() {
                return Err(WalkError::InvalidCpuidOverride);
            }
            out.push(PolicyOp::new(
                policy_kind::CPUID_RANGE,
                start,
                end,
                u64::from(matches!(ov.action, OverrideAction::Native)),
            ));
        }
    }
    Ok((out, emulate_leaves))
}

// ── Value resolution ───────────────────────────────────────────────────── //

/// Resolve a [`ValueOrExpr`] to a concrete `u64`.  Handles the sentinels
/// documented in the config schema:
///   - `"auto:vtom_bit"` → `ctx.vtom_bit as u64`
///   - 4-char ASCII strings (e.g. `"Them"`, `"isCo"`, `"Co\0\0"`) →
///     the little-endian byte-encoded u32.
///   - Plain integers (hex or decimal) from the JSON layer.
pub(super) fn resolve_scalar(v: &ValueOrExpr, ctx: WalkContext) -> Result<u64, WalkError> {
    match v {
        ValueOrExpr::Int(n) => Ok(*n),
        ValueOrExpr::Str(s) => resolve_str(s, ctx),
    }
}

fn resolve_str(s: &str, ctx: WalkContext) -> Result<u64, WalkError> {
    if s == "auto:vtom_bit" {
        return Ok(ctx.vtom_bit as u64);
    }
    if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        return u64::from_str_radix(rest, 16)
            .map_err(|_| WalkError::UnknownExpression(s.to_string()));
    }
    if let Ok(n) = s.parse::<u64>() {
        return Ok(n);
    }
    // 4-char ASCII register signature (used for CoCo detection leaf).  Pad
    // shorter strings with zeros on the right; longer strings error.
    let bytes = s.as_bytes();
    if bytes.len() <= 4 {
        let mut arr = [0u8; 4];
        arr[..bytes.len()].copy_from_slice(bytes);
        return Ok(u32::from_le_bytes(arr) as u64);
    }
    Err(WalkError::UnknownExpression(s.to_string()))
}

fn default_action_value(a: DefaultAction) -> u64 {
    match a {
        DefaultAction::Trap => 0,
        DefaultAction::Native => 1,
    }
}

/// Compute the Native CPUID hypervisor-range segments, split around the
/// emulate leaves supplied.  Returns `(range_lo, range_hi)` pairs
/// (inclusive) for the caller to convert into `CPUID_RANGE` ops.
///
/// Preserves the exact split logic used in `vm_state.rs::ensure_initialized`
/// prior to this refactor (see git history around 2026-07-16).
#[cfg(test)]
fn hypervisor_range_native_segments(
    range_start: u32,
    range_end: u32,
    mut emulate_leaves: Vec<u32>,
) -> Vec<(u32, u32)> {
    emulate_leaves.sort();
    emulate_leaves.dedup();

    let mut out = Vec::new();
    let mut cursor = range_start;
    for leaf in &emulate_leaves {
        let leaf = *leaf;
        if cursor < leaf {
            out.push((cursor, leaf - 1));
        }
        cursor = leaf + 1;
    }
    if cursor <= range_end {
        out.push((cursor, range_end));
    }
    out
}

// ── Regression harness ─────────────────────────────────────────────────── //

#[cfg(test)]
mod tests {
    use super::*;
    use crate::themis::config::{DefaultProfile, ThemisConfig};
    use crate::themis::consts::policy_kind;

    #[test]
    fn cpuid_native_and_trap_ranges_override_vp_tables_without_overlap() {
        let cfg = ThemisConfig::from_json_str(
            r#"{"policies":{"cpuid":{"overrides":[
            {"leaf":1,"subleaf":3,"action":"Native"},
            {"range":[[1,5],[2,2]],"action":"Trap"}
        ]}}}"#,
        )
        .unwrap();
        let (ops, _) = walk_cpuid(&cfg, WalkContext { vtom_bit: 39 }).unwrap();
        assert!(ops.contains(&PolicyOp::new(
            policy_kind::CPUID_RANGE,
            (1 << 32) | 3,
            (1 << 32) | 3,
            1
        )));
        assert!(ops.contains(&PolicyOp::new(
            policy_kind::CPUID_RANGE,
            (1 << 32) | 5,
            (2 << 32) | 2,
            0
        )));
        let table = vec![
            CpuIdEntry {
                function: 1,
                eax: 11,
                ..Default::default()
            },
            CpuIdEntry {
                function: 2,
                eax: 22,
                ..Default::default()
            },
        ];
        let scoped = walk_vp_cpuid(&[table], &ops).unwrap();
        let range_kind = policy_kind::for_vp(policy_kind::CPUID_EMULATE_RANGE, 0).unwrap();
        let emitted = scoped
            .iter()
            .filter(|op| op.kind == range_kind)
            .map(|op| (op.key, op.sub_key))
            .collect::<Vec<_>>();
        assert_eq!(
            emitted,
            vec![
                (1 << 32, (1 << 32) | 2),
                ((1 << 32) | 4, (1 << 32) | 4),
                ((2 << 32) | 3, (2 << 32) | u64::from(u32::MAX)),
            ]
        );
        assert_eq!(
            cpuid_range_segments(0, u64::MAX, &[(4, 8), (u64::MAX, u64::MAX)]),
            vec![(0, 3), (9, u64::MAX - 1)]
        );
    }

    #[test]
    fn cpuid_overrides_reject_overlaps_and_invalid_coordinates() {
        for overrides in [
            r#"[{"range":[[2,0],[1,0]],"action":"Trap"}]"#,
            r#"[{"leaf":1,"range":[[1,0],[1,1]],"action":"Trap"}]"#,
            r#"[{"range":[["0x100000000",0],[1,0]],"action":"Trap"}]"#,
            r#"[{"leaf":1,"action":"Native","eax":1}]"#,
            r#"[{"range":[[1,0],[1,2]],"action":"Trap"},{"leaf":1,"subleaf":1,"action":"Emulate","eax":1}]"#,
        ] {
            let cfg = ThemisConfig::from_json_str(&format!(
                r#"{{"policies":{{"cpuid":{{"overrides":{overrides}}}}}}}"#
            ))
            .unwrap();
            walk_cpuid(&cfg, WalkContext { vtom_bit: 39 }).unwrap_err();
        }
    }

    /// Emit the walker-driven policy stream for the given default profile
    /// and verify it exactly matches the legacy hardcoded stream in
    /// `vm_state.rs` for the "MSR baseline + CPUID Emulate overrides"
    /// slice (runtime injections are excluded — those are still layered
    /// on top by `vm_state.rs`).
    fn baseline_expected_standard() -> Vec<PolicyOp> {
        vec![
            // MSR_DEFAULT = Native
            PolicyOp::new(policy_kind::MSR_DEFAULT, 0, 0, 1),
            // MSR_EMULATE(0x6E0, sub=0, value=0) — matches vm_state.rs:237
            PolicyOp::new(policy_kind::MSR_EMULATE, 0x6E0, 0, 0),
        ]
    }

    fn baseline_expected_confidential(vtom_bit: u32) -> Vec<PolicyOp> {
        let mut v = baseline_expected_standard();
        // CoCo detection leaf, matches vm_state.rs:468-478.
        let coco_leaf: u64 = 0x40000100_u64 << 32;
        let eax = vtom_bit;
        let ebx = u32::from_le_bytes(*b"Them");
        let ecx = u32::from_le_bytes(*b"isCo");
        let edx = u32::from_le_bytes(*b"Co\0\0");
        let word0 = ((eax as u64) << 32) | (ebx as u64);
        let word1 = ((ecx as u64) << 32) | (edx as u64);
        v.push(PolicyOp::new(
            policy_kind::CPUID_EMULATE,
            coco_leaf,
            0,
            word0,
        ));
        v.push(PolicyOp::new(
            policy_kind::CPUID_EMULATE,
            coco_leaf,
            1,
            word1,
        ));
        v
    }

    #[test]
    fn api_permissions_include_doorbells_and_reject_unknown_names() {
        let cfg = ThemisConfig::from_json_str(
            r#"{"policies":{"api":{"allow":["send","attest","CREATE_DOORBELL","notify"]}}}"#,
        )
        .unwrap();
        assert_eq!(
            walk_api(&cfg).unwrap(),
            PolicyOp::new(policy_kind::API_MONITOR, 0, 0, 0xc028)
        );
        let invalid =
            ThemisConfig::from_json_str(r#"{"policies":{"api":{"allow":["notify","NOTFIY"]}}}"#)
                .unwrap();
        assert!(matches!(
            walk_api(&invalid),
            Err(WalkError::UnknownMonitorApi(name)) if name == "NOTFIY"
        ));
    }

    #[test]
    fn api_default_and_explicit_empty_list_differ() {
        let cfg = ThemisConfig::builtin_default(DefaultProfile::Standard).unwrap();
        assert_eq!(walk_api(&cfg).unwrap().value, 0xffff);
        let empty = ThemisConfig::from_json_str(r#"{"policies":{"api":{"allow":[]}}}"#).unwrap();
        assert_eq!(walk_api(&empty).unwrap().value, 0);
    }

    #[test]
    fn standard_default_msr_stream_matches_legacy() {
        let cfg = ThemisConfig::builtin_default(DefaultProfile::Standard).unwrap();
        let ctx = WalkContext { vtom_bit: 0 };
        let ops = walk_msrs(&cfg, ctx).unwrap();
        assert_eq!(ops, baseline_expected_standard());
    }

    #[test]
    fn confidential_default_msr_stream_matches_legacy() {
        let cfg = ThemisConfig::builtin_default(DefaultProfile::Confidential).unwrap();
        let ctx = WalkContext { vtom_bit: 47 };
        let ops = walk_msrs(&cfg, ctx).unwrap();
        // MSR baseline is identical between standard and confidential.
        assert_eq!(ops, baseline_expected_standard());
    }

    #[test]
    fn confidential_default_cpuid_emulate_stream_matches_legacy() {
        let cfg = ThemisConfig::builtin_default(DefaultProfile::Confidential).unwrap();
        let ctx = WalkContext { vtom_bit: 47 };
        let (ops, leaves) = walk_cpuid(&cfg, ctx).unwrap();
        let expected: Vec<PolicyOp> = baseline_expected_confidential(47)
            .into_iter()
            .filter(|op| op.kind == policy_kind::CPUID_EMULATE)
            .collect();
        assert_eq!(ops[0], PolicyOp::new(policy_kind::CPUID_DEFAULT, 0, 0, 1));
        assert_eq!(ops[1..], expected);
        assert_eq!(leaves, vec![0x40000100]);
    }

    #[test]
    fn standard_default_has_no_cpuid_emulates() {
        let cfg = ThemisConfig::builtin_default(DefaultProfile::Standard).unwrap();
        let ctx = WalkContext { vtom_bit: 0 };
        let (ops, leaves) = walk_cpuid(&cfg, ctx).unwrap();
        assert_eq!(ops, [PolicyOp::new(policy_kind::CPUID_DEFAULT, 0, 0, 1)]);
        assert!(leaves.is_empty());
        let mut trapped = cfg;
        trapped.policies.cpuid.default = DefaultAction::Trap;
        assert_eq!(
            walk_cpuid(&trapped, ctx).unwrap().0,
            [PolicyOp::new(policy_kind::CPUID_DEFAULT, 0, 0, 0)]
        );
    }

    #[test]
    fn confidential_dom1_handles_cr_access_locally_without_parent_register_access() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../themis/policies/dom1-confidential.json");
        let cfg = ThemisConfig::load_from_file(&path).unwrap();
        let ops = walk_exits(&cfg).unwrap();
        let reason = u64::from(vmx_exit_reasons::CR_ACCESS);
        let cr_ops: Vec<_> = ops.iter().filter(|op| op.key == reason).copied().collect();
        let mut expected = vec![PolicyOp::new(policy_kind::EXIT_REASON_TRAP, reason, 0, 0)];
        for kind in [
            policy_kind::EXIT_REASON_REG_READ_SET,
            policy_kind::EXIT_REASON_REG_WRITE_SET,
        ] {
            for word in 0..3 {
                expected.push(PolicyOp::new(kind, reason, word, 0));
            }
        }
        assert_eq!(cr_ops, expected);
        assert_eq!(
            ops[0],
            PolicyOp::new(policy_kind::DEFAULT_EXIT_TRAP, 0, 0, 1)
        );
        for kind in [
            policy_kind::DEFAULT_EXIT_REG_READ_SET,
            policy_kind::DEFAULT_EXIT_REG_WRITE_SET,
        ] {
            for word in 0..3 {
                assert!(ops.contains(&PolicyOp::new(kind, 0, word, 0)));
            }
        }
    }

    #[test]
    fn no_exit_default_emits_local_exit_policy() {
        let cfg = ThemisConfig::from_json_str(
            r#"{
                "policies": {
                    "exits": {
                        "default": {
                            "trap": false,
                            "read_set": "ALL",
                            "write_set": "ALL"
                        }
                    }
                }
            }"#,
        )
        .unwrap();

        assert_eq!(
            walk_exits(&cfg).unwrap(),
            vec![
                PolicyOp::new(policy_kind::DEFAULT_EXIT_TRAP, 0, 0, 0),
                PolicyOp::new(policy_kind::DEFAULT_EXIT_REG_READ_SET, 0, 0, u64::MAX),
                PolicyOp::new(policy_kind::DEFAULT_EXIT_REG_READ_SET, 0, 1, u64::MAX),
                PolicyOp::new(policy_kind::DEFAULT_EXIT_REG_READ_SET, 0, 2, u64::MAX),
                PolicyOp::new(policy_kind::DEFAULT_EXIT_REG_WRITE_SET, 0, 0, u64::MAX),
                PolicyOp::new(policy_kind::DEFAULT_EXIT_REG_WRITE_SET, 0, 1, u64::MAX),
                PolicyOp::new(policy_kind::DEFAULT_EXIT_REG_WRITE_SET, 0, 2, u64::MAX),
            ]
        );
    }

    #[test]
    fn pv_returns_never_grant_parent_register_access() {
        let mut cfg = ThemisConfig::from_json_str(include_str!(
            "../../../../eunomia/policies/pv/standard.json"
        ))
        .unwrap();
        let ops = walk_exits(&cfg).unwrap();
        let reason = themis_abi::synthetic_exits::THEMIS_EXIT_YIELD;
        let reason_ops = ops
            .iter()
            .filter(|op| op.key == u64::from(reason))
            .collect::<Vec<_>>();
        assert_eq!(reason_ops.len(), 7);
        assert_eq!(reason_ops[0].kind, policy_kind::EXIT_REASON_TRAP);
        assert_eq!(reason_ops[0].value, 1);
        assert!(reason_ops[1..].iter().all(|op| op.value == 0));
        let override_config = format!(
            r#"{{"policies":{{"exits":{{"overrides":[{{
              "reason":{reason},"trap":true,"read_set":["RAX"],"write_set":"NONE"
            }}]}}}}}}"#
        );
        cfg.policies.exits = ThemisConfig::from_json_str(&override_config)
            .unwrap()
            .policies
            .exits;
        assert!(matches!(walk_exits(&cfg), Err(WalkError::UnsafePvExit(_))));
    }

    #[test]
    fn exit_override_emits_trap_and_register_words() {
        let cfg = ThemisConfig::from_json_str(
            r#"{
                "policies": {
                    "exits": {
                        "overrides": [{
                            "reason": "IO_INSTRUCTION",
                            "trap": true,
                            "read_set": ["RAX", "RIP"],
                            "write_set": "NONE"
                        }]
                    }
                }
            }"#,
        )
        .unwrap();

        let ops = walk_exits(&cfg).unwrap();
        assert_eq!(
            ops[0],
            PolicyOp::new(policy_kind::DEFAULT_EXIT_TRAP, 0, 0, 1)
        );
        let ops = &ops[6..];
        assert_eq!(
            ops[1],
            PolicyOp::new(
                policy_kind::EXIT_REASON_TRAP,
                vmx_exit_reasons::IO_INSTRUCTION as u64,
                0,
                1,
            )
        );
        assert_eq!(ops[2].value, 1 | (1 << (VpRegister::Rip as u64)));
        assert_eq!(ops[3].value, 0);
        assert_eq!(ops[4].value, 0);
        assert_eq!(ops[5].value, 0);
        assert_eq!(ops[6].value, 0);
        assert_eq!(ops[7].value, 0);
    }

    #[test]
    fn interrupt_default_emits_complete_policy() {
        let cfg = ThemisConfig::from_json_str(
            r#"{
                "policies": {
                    "interrupts": {
                        "default": {
                            "visibility": "NotReport",
                            "read_set": "NONE",
                            "write_set": "NONE",
                            "injectable": false
                        }
                    }
                }
            }"#,
        )
        .unwrap();

        assert_eq!(
            walk_interrupts(&cfg).unwrap(),
            vec![
                PolicyOp::new(policy_kind::DEFAULT_INTR_VISIBILITY, 0, 0, 2),
                PolicyOp::new(policy_kind::DEFAULT_INTR_REG_READ_SET, 0, 0, 0),
                PolicyOp::new(policy_kind::DEFAULT_INTR_REG_READ_SET, 0, 1, 0),
                PolicyOp::new(policy_kind::DEFAULT_INTR_REG_READ_SET, 0, 2, 0),
                PolicyOp::new(policy_kind::DEFAULT_INTR_REG_WRITE_SET, 0, 0, 0),
                PolicyOp::new(policy_kind::DEFAULT_INTR_REG_WRITE_SET, 0, 1, 0),
                PolicyOp::new(policy_kind::DEFAULT_INTR_REG_WRITE_SET, 0, 2, 0),
                PolicyOp::new(policy_kind::DEFAULT_VECTOR_INJECTABLE, 0, 0, 0),
            ]
        );
    }

    #[test]
    fn vp_cpuid_keeps_distinct_tables_and_configured_overrides() {
        let entry = CpuIdEntry {
            function: 1,
            index: 0,
            flags: 0,
            eax: 7,
            ebx: 0,
            ecx: 8,
            edx: 9,
        };
        let mut ap = entry;
        ap.ebx = 1 << 24;
        let configured = [
            PolicyOp::new(policy_kind::CPUID_EMULATE, 1 << 32, 0, 42),
            PolicyOp::new(policy_kind::CPUID_EMULATE, 1 << 32, 1, 43),
        ];
        let ops = walk_vp_cpuid(&[vec![entry], vec![ap]], &configured).unwrap();
        for vp in 0..2 {
            let base = vp as usize * 4;
            let kind = policy_kind::for_vp(policy_kind::CPUID_EMULATE, vp).unwrap();
            let range_kind = policy_kind::for_vp(policy_kind::CPUID_EMULATE_RANGE, vp).unwrap();
            assert_eq!(
                ops[base],
                PolicyOp::new(
                    range_kind,
                    (1 << 32) | 1,
                    (1 << 32) | u64::from(u32::MAX),
                    (7 << 32) | (u64::from(vp) << 24)
                )
            );
            assert_eq!(
                ops[base + 1],
                PolicyOp::new(
                    policy_kind::with_emulate_word(range_kind, 1).unwrap(),
                    (1 << 32) | 1,
                    (1 << 32) | u64::from(u32::MAX),
                    (8 << 32) | 9
                )
            );
            assert_eq!(
                ops[base + 2],
                PolicyOp {
                    kind,
                    ..configured[0]
                }
            );
            assert_eq!(
                ops[base + 3],
                PolicyOp {
                    kind,
                    ..configured[1]
                }
            );
        }
        let too_many = vec![Vec::new(); themis_abi::MAX_VPS_PER_DOMAIN as usize + 1];
        assert!(matches!(
            walk_vp_cpuid(&too_many, &[]),
            Err(WalkError::InvalidVpIndex(_))
        ));
    }

    #[test]
    fn vp_cpuid_covers_ignored_indices_and_sparse_indexed_leaves() {
        let ordinary = CpuIdEntry {
            function: 0,
            eax: 7,
            ebx: 8,
            ecx: 9,
            edx: 10,
            ..CpuIdEntry::default()
        };
        let mut other_vp = ordinary;
        other_vp.ebx = 42;
        let indexed = |index, eax| CpuIdEntry {
            function: 7,
            index,
            flags: CPUID_FLAG_VALID_INDEX,
            eax,
            ..CpuIdEntry::default()
        };
        let ops = walk_vp_cpuid(
            &[
                vec![ordinary, indexed(0, 3), indexed(3, 99)],
                vec![other_vp],
            ],
            &[],
        )
        .unwrap();
        let reply = |vp, leaf, subleaf| {
            let key = (u64::from(leaf) << 32) | u64::from(subleaf);
            let mut words = [None; 2];
            for op in &ops {
                let (kind, scope, word) = policy_kind::decode(op.kind).unwrap();
                if scope == Some(vp)
                    && kind == policy_kind::CPUID_EMULATE_RANGE
                    && op.key <= key
                    && key <= op.sub_key
                {
                    assert!(
                        words[word as usize].replace(op.value).is_none(),
                        "overlapping emitted ranges"
                    );
                }
            }
            words
        };
        for vp in 0..2 {
            for subleaf in [0, 0xc35380, u32::MAX] {
                assert_eq!(
                    reply(vp, 0u32, subleaf),
                    [
                        Some((7 << 32) | if vp == 0 { 8 } else { 42 }),
                        Some((9 << 32) | 10)
                    ]
                );
            }
        }
        for (subleaf, eax) in [(0, 3), (1, 0), (2, 0), (3, 99), (4, 0), (u32::MAX, 0)] {
            assert_eq!(reply(0, 7, subleaf), [Some(eax << 32), Some(0)]);
        }
        assert_eq!(reply(0, 8, 0), [None, None]);
    }

    #[test]
    fn vp_cpuid_rejects_ambiguous_tables_instead_of_guessing() {
        let entry = CpuIdEntry {
            function: 1,
            ..CpuIdEntry::default()
        };
        for table in [
            vec![entry, entry],
            vec![entry, CpuIdEntry { index: 1, ..entry }],
            vec![
                entry,
                CpuIdEntry {
                    flags: CPUID_FLAG_VALID_INDEX,
                    index: 1,
                    ..entry
                },
            ],
            vec![CpuIdEntry { flags: 2, ..entry }],
            vec![
                CpuIdEntry {
                    flags: CPUID_FLAG_VALID_INDEX,
                    ..entry
                };
                2
            ],
        ] {
            assert!(matches!(
                walk_vp_cpuid(&[table], &[]),
                Err(WalkError::InvalidCpuidTable { vp: 0, leaf: 1 })
            ));
        }
    }

    #[test]
    fn interrupt_override_emits_visibility_regsets_and_injectable() {
        let cfg = ThemisConfig::from_json_str(
            r#"{
                "policies": {
                    "interrupts": {
                        "default": { "visibility": "NotReport", "injectable": false },
                        "overrides": [{
                            "vector": 236,
                            "visibility": "NotReport",
                            "read_set": "ALL",
                            "write_set": "NONE",
                            "injectable": true
                        }]
                    }
                }
            }"#,
        )
        .unwrap();

        let ops = walk_interrupts(&cfg).unwrap();
        assert_eq!(
            ops[0],
            PolicyOp::new(policy_kind::DEFAULT_INTR_VISIBILITY, 0, 0, 2)
        );
        let ops = &ops[7..];
        assert_eq!(
            ops[1],
            PolicyOp::new(policy_kind::VECTOR_VISIBILITY, 236, 0, 2)
        );
        // read_set = ALL -> 3 words of u64::MAX.
        assert_eq!(
            ops[2],
            PolicyOp::new(policy_kind::VECTOR_REG_READ_SET, 236, 0, u64::MAX)
        );
        assert_eq!(
            ops[3],
            PolicyOp::new(policy_kind::VECTOR_REG_READ_SET, 236, 1, u64::MAX)
        );
        assert_eq!(
            ops[4],
            PolicyOp::new(policy_kind::VECTOR_REG_READ_SET, 236, 2, u64::MAX)
        );
        // write_set = NONE -> 3 words of 0.
        assert_eq!(
            ops[5],
            PolicyOp::new(policy_kind::VECTOR_REG_WRITE_SET, 236, 0, 0)
        );
        assert_eq!(
            ops[6],
            PolicyOp::new(policy_kind::VECTOR_REG_WRITE_SET, 236, 1, 0)
        );
        assert_eq!(
            ops[7],
            PolicyOp::new(policy_kind::VECTOR_REG_WRITE_SET, 236, 2, 0)
        );
        // injectable=true, independent of the (also NotReport) visibility.
        assert_eq!(
            ops[8],
            PolicyOp::new(policy_kind::VECTOR_INJECTABLE, 236, 0, 1)
        );
    }

    #[test]
    fn hypervisor_range_splits_match_legacy_algorithm() {
        // Reproduce vm_state.rs:425-465 splitting logic.
        // Non-confidential with 1 ivshmem device:
        let segs = hypervisor_range_native_segments(0x40000000, 0x4FFFFFFF, vec![0x40000004]);
        assert_eq!(
            segs,
            vec![(0x40000000, 0x40000003), (0x40000005, 0x4FFFFFFF)]
        );

        // Confidential with 1 ivshmem device (2 emulate leaves).
        let segs =
            hypervisor_range_native_segments(0x40000000, 0x4FFFFFFF, vec![0x40000004, 0x40000100]);
        assert_eq!(
            segs,
            vec![
                (0x40000000, 0x40000003),
                (0x40000005, 0x400000FF),
                (0x40000101, 0x4FFFFFFF),
            ]
        );

        // No emulate leaves.
        let segs = hypervisor_range_native_segments(0x40000000, 0x4FFFFFFF, vec![]);
        assert_eq!(segs, vec![(0x40000000, 0x4FFFFFFF)]);
    }

    #[test]
    fn resolve_auto_vtom_bit() {
        let ctx = WalkContext { vtom_bit: 47 };
        assert_eq!(
            resolve_scalar(&ValueOrExpr::Str("auto:vtom_bit".to_string()), ctx).unwrap(),
            47
        );
    }

    #[test]
    fn resolve_ascii_signature() {
        let ctx = WalkContext { vtom_bit: 0 };
        assert_eq!(
            resolve_scalar(&ValueOrExpr::Str("Them".to_string()), ctx).unwrap(),
            u32::from_le_bytes(*b"Them") as u64
        );
        assert_eq!(
            resolve_scalar(&ValueOrExpr::Str("Co\0\0".to_string()), ctx).unwrap(),
            u32::from_le_bytes(*b"Co\0\0") as u64
        );
    }

    #[test]
    fn resolve_hex_and_dec() {
        let ctx = WalkContext { vtom_bit: 0 };
        assert_eq!(
            resolve_scalar(&ValueOrExpr::Str("0xdeadbeef".to_string()), ctx).unwrap(),
            0xdeadbeef
        );
        assert_eq!(
            resolve_scalar(&ValueOrExpr::Str("12345".to_string()), ctx).unwrap(),
            12345
        );
        assert_eq!(
            resolve_scalar(&ValueOrExpr::Int(0xcafe), ctx).unwrap(),
            0xcafe
        );
    }
}
