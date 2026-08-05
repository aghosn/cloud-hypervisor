//! Walks a [`ThemisConfig`] and emits `THHV_SET_POLICY` tuples in the
//! order the seal path (`vm_state.rs::ensure_msr_policy_pushed` +
//! `ensure_initialized`) expects.
//!
//! Design constraint (see `docs/chv-themis-config.md` v0.4 and the
//! `chv-themis-config-apply` todo): when a caller loads the bundled
//! default profile matching its `confidential` flag, the emitted
//! `(kind, key, sub_key, value)` stream MUST be byte-identical to the
//! previously-hardcoded stream in `vm_state.rs`.  A dedicated
//! regression test in this file asserts that invariant.
//!
//! Only the pieces that appear in the default JSON profiles today are
//! driven from the config here:
//!
//!   - MSR default action  (`msrs.default`)
//!   - MSR overrides       (`msrs.overrides`) — Native / Trap / Emulate
//!   - CPUID Emulate overrides (`cpuid.overrides` with `action: Emulate`)
//!   - CPUID hypervisor-range Native  (`cpuid.hypervisor_range_native`)
//!
//! The runtime-dependent injections (x2APIC MSR block, per-VM CPUID
//! entries, ivshmem BAR CPUID leaves, ranges split around emulate
//! leaves) are still layered on top by `vm_state.rs` — this walker
//! provides the helpers they need to compose the final stream.

use themis_abi::regs::VpRegister;
use themis_abi::vmx_exit_reasons;

use super::config::{
    DefaultAction, ExitReason, MsrOverride, OverrideAction, RegSet, RegSetSymbolic, ThemisConfig,
    ValueOrExpr, Visibility,
};
use super::consts::policy_kind;

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
        Self { kind, key, sub_key, value }
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
    #[error("cpuid override uses a range with action Emulate (unsupported: emulate needs per-leaf values)")]
    EmulateOverRange,
    #[error("msr override has neither `msr` nor `range` set")]
    MsrOverrideMissingTarget,
    #[error("msr range [{lo:#x},{hi:#x}] with action Emulate — Emulate requires a single MSR")]
    EmulateOverMsrRange { lo: u32, hi: u32 },
    #[error("unknown VMEXIT reason: {0:?}")]
    UnknownExitReason(String),
    #[error("unknown VP register name in register set: {0:?}")]
    UnknownRegister(String),
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
/// Register-access bitmaps and `injectable` are only meaningful as
/// per-vector overrides today; the wire ABI has no separate "default
/// read/write set" or "default injectable" op, so `cfg.policies.interrupts
/// .default`'s read_set/write_set/injectable fields are only applied when a
/// vector override is also present for the same vector's other fields (this
/// mirrors the pre-existing engine default: `VectorPolicy::default_report()`).
pub(super) fn walk_interrupts(cfg: &ThemisConfig) -> Result<Vec<PolicyOp>, WalkError> {
    let mut out = vec![PolicyOp::new(
        policy_kind::DEFAULT_INTR_VISIBILITY,
        0,
        0,
        visibility_wire_value(cfg.policies.interrupts.default.visibility),
    )];

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
/// The wire API exposes the default trap bit plus complete per-reason
/// overrides. Default read/write sets are not separately addressable by the
/// current ABI; they remain at the engine default (`ALL`).
pub(super) fn walk_exits(cfg: &ThemisConfig) -> Result<Vec<PolicyOp>, WalkError> {
    let mut out = vec![PolicyOp::new(
        policy_kind::DEFAULT_EXIT_TRAP,
        0,
        0,
        u64::from(cfg.policies.exits.default.trap),
    )];

    for ov in &cfg.policies.exits.overrides {
        let reason = resolve_exit_reason(&ov.reason)?;
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

    Ok(out)
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
        RegSet::Symbolic(RegSetSymbolic::ALL) => Ok([u64::MAX; 3]),
        RegSet::Symbolic(RegSetSymbolic::NONE) => Ok([0; 3]),
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
pub(super) fn walk_msrs(
    cfg: &ThemisConfig,
    ctx: WalkContext,
) -> Result<Vec<PolicyOp>, WalkError> {
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
                (None, Some([lo, hi])) => {
                    return Err(WalkError::EmulateOverMsrRange { lo, hi })
                }
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
                out.push(PolicyOp::new(policy_kind::MSR_EMULATE, msr as u64, 0, value));
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

/// Emit the CPUID Emulate overrides declared in `cfg` (only Emulate — the
/// Native/Trap directions today are governed by hypervisor-range Native
/// installation which the caller handles separately, since it interacts
/// with runtime-injected per-VM emulate leaves).
///
/// Returns the emitted ops plus the set of `(leaf)` numbers that should
/// be excluded from any Native hypervisor-range push.
pub(super) fn walk_cpuid_emulates(
    cfg: &ThemisConfig,
    ctx: WalkContext,
) -> Result<(Vec<PolicyOp>, Vec<u32>), WalkError> {
    let mut out = Vec::new();
    let mut emulate_leaves = Vec::new();

    for ov in &cfg.policies.cpuid.overrides {
        if matches!(ov.action, OverrideAction::Emulate) {
            let (leaf, subleaf) = match (ov.leaf, ov.subleaf, &ov.range) {
                (Some(l), s, None) => (l, s.unwrap_or(0)),
                (None, _, Some(_)) => return Err(WalkError::EmulateOverRange),
                _ => {
                    return Err(WalkError::MissingCpuidValues {
                        leaf: 0,
                        sub: 0,
                    })
                }
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
                return Err(WalkError::MissingCpuidValues {
                    leaf,
                    sub: subleaf,
                });
            }

            let eax = eax.unwrap_or(0) as u32;
            let ebx = ebx.unwrap_or(0) as u32;
            let ecx = ecx.unwrap_or(0) as u32;
            let edx = edx.unwrap_or(0) as u32;

            let key = ((leaf as u64) << 32) | (subleaf as u64);
            let word0 = ((eax as u64) << 32) | (ebx as u64);
            let word1 = ((ecx as u64) << 32) | (edx as u64);
            out.push(PolicyOp::new(policy_kind::CPUID_EMULATE, key, 0, word0));
            out.push(PolicyOp::new(policy_kind::CPUID_EMULATE, key, 1, word1));
            emulate_leaves.push(leaf);
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
pub(super) fn resolve_scalar(
    v: &ValueOrExpr,
    ctx: WalkContext,
) -> Result<u64, WalkError> {
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
pub(super) fn hypervisor_range_native_segments(
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
        v.push(PolicyOp::new(policy_kind::CPUID_EMULATE, coco_leaf, 0, word0));
        v.push(PolicyOp::new(policy_kind::CPUID_EMULATE, coco_leaf, 1, word1));
        v
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
        let (ops, leaves) = walk_cpuid_emulates(&cfg, ctx).unwrap();
        let expected: Vec<PolicyOp> = baseline_expected_confidential(47)
            .into_iter()
            .filter(|op| op.kind == policy_kind::CPUID_EMULATE)
            .collect();
        assert_eq!(ops, expected);
        assert_eq!(leaves, vec![0x40000100]);
    }

    #[test]
    fn standard_default_has_no_cpuid_emulates() {
        let cfg = ThemisConfig::builtin_default(DefaultProfile::Standard).unwrap();
        let ctx = WalkContext { vtom_bit: 0 };
        let (ops, leaves) = walk_cpuid_emulates(&cfg, ctx).unwrap();
        assert!(ops.is_empty());
        assert!(leaves.is_empty());
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
            vec![PolicyOp::new(policy_kind::DEFAULT_EXIT_TRAP, 0, 0, 0)]
        );
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
    fn interrupt_default_visibility_only() {
        let cfg = ThemisConfig::from_json_str(
            r#"{
                "policies": {
                    "interrupts": {
                        "default": { "visibility": "NotReport" }
                    }
                }
            }"#,
        )
        .unwrap();

        assert_eq!(
            walk_interrupts(&cfg).unwrap(),
            vec![PolicyOp::new(policy_kind::DEFAULT_INTR_VISIBILITY, 0, 0, 2)]
        );
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
        assert_eq!(
            ops[1],
            PolicyOp::new(policy_kind::VECTOR_VISIBILITY, 236, 0, 2)
        );
        // read_set = ALL -> 3 words of u64::MAX.
        assert_eq!(ops[2], PolicyOp::new(policy_kind::VECTOR_REG_READ_SET, 236, 0, u64::MAX));
        assert_eq!(ops[3], PolicyOp::new(policy_kind::VECTOR_REG_READ_SET, 236, 1, u64::MAX));
        assert_eq!(ops[4], PolicyOp::new(policy_kind::VECTOR_REG_READ_SET, 236, 2, u64::MAX));
        // write_set = NONE -> 3 words of 0.
        assert_eq!(ops[5], PolicyOp::new(policy_kind::VECTOR_REG_WRITE_SET, 236, 0, 0));
        assert_eq!(ops[6], PolicyOp::new(policy_kind::VECTOR_REG_WRITE_SET, 236, 1, 0));
        assert_eq!(ops[7], PolicyOp::new(policy_kind::VECTOR_REG_WRITE_SET, 236, 2, 0));
        // injectable=true, independent of the (also NotReport) visibility.
        assert_eq!(ops[8], PolicyOp::new(policy_kind::VECTOR_INJECTABLE, 236, 0, 1));
    }

    #[test]
    fn hypervisor_range_splits_match_legacy_algorithm() {
        // Reproduce vm_state.rs:425-465 splitting logic.
        // Non-confidential with 1 ivshmem device:
        let segs = hypervisor_range_native_segments(
            0x40000000,
            0x4FFFFFFF,
            vec![0x40000004],
        );
        assert_eq!(
            segs,
            vec![(0x40000000, 0x40000003), (0x40000005, 0x4FFFFFFF)]
        );

        // Confidential with 1 ivshmem device (2 emulate leaves).
        let segs = hypervisor_range_native_segments(
            0x40000000,
            0x4FFFFFFF,
            vec![0x40000004, 0x40000100],
        );
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
