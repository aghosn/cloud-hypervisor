//! User-facing Themis policy configuration.
//!
//! Deserialised from a JSON file supplied via `--themis-config` (see
//! `docs/chv-themis-config.md`, schema v0.4).  The top-level shape is:
//!
//! ```jsonc
//! {
//!   "extends": "standard" | "confidential" | <path>,
//!   "general":  { name, confidential, vtom_bit, shared_regions },
//!   "policies": { cores, api, interrupts, exits, msrs, cpuid },
//!   "comm":     { ivshmem: [...] }
//! }
//! ```
//!
//! `policies` mirrors `capa-engine::domain::DomainPolicy` 1-to-1.
//!
//! This module owns only the *parsed* representation.  Applying the config
//! to actual THHV_SET_POLICY calls is done in
//! `cloud-hypervisor/hypervisor/src/themis/vm_state.rs` and is scheduled
//! for a later task (`chv-themis-config-apply`).  Reconciliation with the
//! CLI `--ivshmem` list happens in `vmm/src/config.rs` and is scheduled
//! for `chv-themis-config-ivshmem-merge`.

use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Embedded default profiles ───────────────────────────────────────────── //

/// Built-in default policy applied when the user provides no
/// `--themis-config` AND `--platform confidential=off` (or unset).
/// Extracted from the hardcoded values previously in
/// `vm_state.rs::ensure_msr_policy_pushed` and the seal-time CPUID push;
/// the runtime-conditional bits (x2APIC MSR ranges, per-VM CPUID entries,
/// ivshmem BARs) are still layered on top at seal time.
pub const STANDARD_DEFAULT_JSON: &str = include_str!("defaults/standard.json");

/// Same but for `--platform confidential=on`.  Adds the CoCo detection
/// CPUID leaf (0x40000100) with an `"auto:vtom_bit"` sentinel.
pub const CONFIDENTIAL_DEFAULT_JSON: &str = include_str!("defaults/confidential.json");

/// Dev override paths installed by [`set_default_override`].  When non-`None`,
/// [`ThemisConfig::builtin_default`] loads the profile from disk instead of
/// the embedded `include_str!` copy.  Off by default; toggled via
/// `--themis-default-override <path>` in the CLI.
struct DefaultOverrides {
    standard: Option<PathBuf>,
    confidential: Option<PathBuf>,
}

static DEFAULT_OVERRIDES: OnceLock<RwLock<DefaultOverrides>> = OnceLock::new();

fn overrides() -> &'static RwLock<DefaultOverrides> {
    DEFAULT_OVERRIDES.get_or_init(|| {
        RwLock::new(DefaultOverrides {
            standard: None,
            confidential: None,
        })
    })
}

/// Install a runtime override for the given built-in default profile.
/// Subsequent calls to [`ThemisConfig::builtin_default`] and `extends`
/// resolution will read the profile from `path` instead of the embedded
/// JSON.  Intended for dev iteration; production runs use the embedded
/// copy.  Pass `None` to clear.
pub fn set_default_override(profile: DefaultProfile, path: Option<PathBuf>) {
    let mut o = overrides().write().unwrap();
    match profile {
        DefaultProfile::Standard => o.standard = path,
        DefaultProfile::Confidential => o.confidential = path,
    }
}

/// Which built-in default profile to load.  Selected by CHV based on
/// `--platform confidential=on/off` when no `--themis-config` is given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefaultProfile {
    Standard,
    Confidential,
}

impl DefaultProfile {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "standard" => Some(DefaultProfile::Standard),
            "confidential" => Some(DefaultProfile::Confidential),
            _ => None,
        }
    }

    fn embedded_json(self) -> &'static str {
        match self {
            DefaultProfile::Standard => STANDARD_DEFAULT_JSON,
            DefaultProfile::Confidential => CONFIDENTIAL_DEFAULT_JSON,
        }
    }

    fn override_path(self) -> Option<PathBuf> {
        let o = overrides().read().unwrap();
        match self {
            DefaultProfile::Standard => o.standard.clone(),
            DefaultProfile::Confidential => o.confidential.clone(),
        }
    }
}

// ── Errors ──────────────────────────────────────────────────────────────── //

#[derive(Debug, Error)]
pub enum ThemisConfigError {
    #[error("failed to read themis config file {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse themis config file {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error(
        "themis config 'extends' name {name:?} is not a known built-in \
         (expected \"standard\", \"confidential\", or a path)"
    )]
    ExtendsUnknownBuiltin { name: String },

    #[error("themis config 'extends' chain deeper than one level (base {base} itself extends)")]
    ExtendsChainTooDeep { base: String },

    #[error("themis config validation: {0}")]
    Validation(String),
}

pub type Result<T> = std::result::Result<T, ThemisConfigError>;

// ── Root ────────────────────────────────────────────────────────────────── //

/// Complete user-facing Themis policy configuration.
///
/// Loaded from disk with [`ThemisConfig::load_from_file`], which resolves any
/// `extends` field by shallow-merging the base's sections beneath the
/// caller's own.  Sections present in the caller replace the base wholesale;
/// sections absent in the caller are inherited verbatim.  Only one level of
/// `extends` chaining is permitted.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ThemisConfig {
    /// Optional inheritance from a built-in default or another file.
    /// Built-in names: `"standard"`, `"confidential"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extends: Option<String>,

    #[serde(default)]
    pub general: GeneralConfig,

    #[serde(default)]
    pub policies: PoliciesConfig,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comm: Option<CommConfig>,
}

impl ThemisConfig {
    /// Load a config from disk and resolve its `extends` reference.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path).map_err(|source| ThemisConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let mut cfg: ThemisConfig =
            serde_json::from_str(&raw).map_err(|source| ThemisConfigError::Parse {
                path: path.to_path_buf(),
                source,
            })?;
        cfg.resolve_extends(path)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse a config from a JSON string (no extends resolution, no I/O).
    /// Primarily for tests and API consumers.
    pub fn from_json_str(s: &str) -> Result<Self> {
        let cfg: ThemisConfig =
            serde_json::from_str(s).map_err(|source| ThemisConfigError::Parse {
                path: PathBuf::from("<inline>"),
                source,
            })?;
        Ok(cfg)
    }

    /// Load a built-in default profile.  Reads either the embedded JSON
    /// (via `include_str!`) or a runtime override installed with
    /// [`set_default_override`].
    ///
    /// The returned config has any `extends` field cleared and is validated,
    /// but `auto:*` sentinels are preserved verbatim (resolved at apply
    /// time by `chv-themis-config-apply`).
    pub fn builtin_default(profile: DefaultProfile) -> Result<Self> {
        let cfg = match profile.override_path() {
            Some(path) => Self::load_from_file(path)?,
            None => {
                let cfg = Self::from_json_str(profile.embedded_json())?;
                // Built-in JSONs must not chain extends: enforce.
                if cfg.extends.is_some() {
                    return Err(ThemisConfigError::ExtendsChainTooDeep {
                        base: format!("embedded {profile:?}"),
                    });
                }
                cfg.validate()?;
                cfg
            }
        };
        Ok(cfg)
    }

    fn resolve_extends(&mut self, caller_path: &Path) -> Result<()> {
        let extends = match self.extends.take() {
            Some(e) => e,
            None => return Ok(()),
        };

        // Built-in name?
        if let Some(profile) = DefaultProfile::from_name(&extends) {
            let base = Self::builtin_default(profile)?;
            self.merge_over(base);
            return Ok(());
        }

        // Otherwise treat as a path.  Reject anything that "looks like" a
        // built-in name a user might have typo'd (no path separator, no
        // extension) — this is friendlier than a Read error.
        let looks_like_bare_name = !extends.contains('/')
            && !extends.contains('\\')
            && !extends.contains('.');
        if looks_like_bare_name {
            return Err(ThemisConfigError::ExtendsUnknownBuiltin { name: extends });
        }

        let path = PathBuf::from(&extends);
        let base_path = if path.is_absolute() {
            path
        } else {
            caller_path
                .parent()
                .map(|d| d.join(&path))
                .unwrap_or(path)
        };

        let base_raw =
            std::fs::read_to_string(&base_path).map_err(|source| ThemisConfigError::Read {
                path: base_path.clone(),
                source,
            })?;
        let base: ThemisConfig =
            serde_json::from_str(&base_raw).map_err(|source| ThemisConfigError::Parse {
                path: base_path.clone(),
                source,
            })?;

        if base.extends.is_some() {
            return Err(ThemisConfigError::ExtendsChainTooDeep {
                base: base_path.display().to_string(),
            });
        }

        self.merge_over(base);
        Ok(())
    }

    /// Shallow-merge `self` (caller) OVER `base`: sections present in self
    /// win wholesale; sections absent in self are inherited from base.
    fn merge_over(&mut self, base: ThemisConfig) {
        // `general`: whole-section replacement when caller provided a non-default.
        // We approximate "user supplied this" by comparing against Default; users
        // who genuinely want default values should just not extend.
        if self.general == GeneralConfig::default() {
            self.general = base.general;
        }
        if self.policies == PoliciesConfig::default() {
            self.policies = base.policies;
        }
        if self.comm.is_none() {
            self.comm = base.comm;
        }
    }

    /// Cheap sanity checks that don't require touching the runtime.
    fn validate(&self) -> Result<()> {
        // Reject overlapping MSR overrides at the same time.
        // TODO(chv-themis-config-apply): stronger validation once we know the
        // full apply semantics.
        for pair in self.policies.msrs.overrides.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            if a.matches_overlap(b) {
                return Err(ThemisConfigError::Validation(format!(
                    "MSR overrides overlap: {a:?} and {b:?}"
                )));
            }
        }
        Ok(())
    }
}

// ── General ─────────────────────────────────────────────────────────────── //

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(default)]
    pub confidential: bool,

    /// `null` (or absent) means "auto-derive from CPUID leaf 0x80000008".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vtom_bit: Option<u8>,

    /// Confidential-mode plaintext carve-outs (EBDA/ACPI/etc).  Empty for
    /// non-confidential guests.  NOT ivshmem — see docs/chv-themis-config.md.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shared_regions: Vec<SharedRegion>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SharedRegion {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Guest physical address.  Accepts decimal or `"0x..."` string.
    #[serde(with = "hex_or_dec_u64")]
    pub gpa: u64,

    /// Region size in bytes.  Accepts decimal or `"0x..."` string.
    #[serde(with = "hex_or_dec_u64")]
    pub size: u64,
}

// ── Policies (mirrors DomainPolicy) ─────────────────────────────────────── //

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PoliciesConfig {
    #[serde(default)]
    pub cores: CoresConfig,

    #[serde(default)]
    pub api: ApiConfig,

    #[serde(default)]
    pub interrupts: InterruptsConfig,

    #[serde(default)]
    pub exits: ExitsConfig,

    #[serde(default)]
    pub msrs: MsrsConfig,

    #[serde(default)]
    pub cpuid: CpuidConfig,
}

// ── policies.cores ──────────────────────────────────────────────────────── //

/// Allowed physical cores.  Either an explicit list or a bitmask.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CoresConfig {
    /// Explicit list of allowed core indices (0-based).  Mutually exclusive
    /// with `mask`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed: Option<Vec<u32>>,

    /// Bitmap of allowed cores (bit i = core i).  Mutually exclusive with
    /// `allowed`.
    #[serde(default, skip_serializing_if = "Option::is_none", with = "hex_or_dec_u64_opt")]
    pub mask: Option<u64>,
}

impl Default for CoresConfig {
    fn default() -> Self {
        Self {
            allowed: None,
            mask: None,
        }
    }
}

// ── policies.api ────────────────────────────────────────────────────────── //

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ApiConfig {
    /// Either `"ALL"` or an explicit list of monitor API names
    /// (matches variants of `capa-engine::MonitorAPI`).
    #[serde(default = "ApiConfig::default_allow")]
    pub allow: ApiAllow,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            allow: ApiConfig::default_allow(),
        }
    }
}

impl ApiConfig {
    fn default_allow() -> ApiAllow {
        ApiAllow::All
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ApiAllow {
    #[serde(with = "api_all_marker")]
    All,
    List(Vec<String>),
}

/// serde helper: accept the literal string `"ALL"` for `ApiAllow::All`.
mod api_all_marker {
    use serde::{de::Error as _, Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str("ALL")
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<(), D::Error> {
        let s = <&str>::deserialize(d)?;
        if s == "ALL" {
            Ok(())
        } else {
            Err(D::Error::custom(format!(
                "expected \"ALL\" or a list, got {s:?}"
            )))
        }
    }
}

// ── policies.interrupts ─────────────────────────────────────────────────── //

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InterruptsConfig {
    #[serde(default)]
    pub default: VectorPolicyConfig,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overrides: Vec<VectorOverride>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VectorPolicyConfig {
    #[serde(default = "VectorPolicyConfig::default_visibility")]
    pub visibility: Visibility,

    #[serde(default = "RegSet::all")]
    pub read_set: RegSet,

    #[serde(default = "RegSet::all")]
    pub write_set: RegSet,
}

impl Default for VectorPolicyConfig {
    fn default() -> Self {
        Self {
            visibility: VectorPolicyConfig::default_visibility(),
            read_set: RegSet::all(),
            write_set: RegSet::all(),
        }
    }
}

impl VectorPolicyConfig {
    fn default_visibility() -> Visibility {
        Visibility::Report
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum Visibility {
    Deliver,
    Report,
    Suppress,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VectorOverride {
    pub vector: u8,
    #[serde(flatten)]
    pub policy: VectorPolicyConfig,
}

// ── policies.exits ──────────────────────────────────────────────────────── //

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExitsConfig {
    #[serde(default)]
    pub default: ExitAction,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overrides: Vec<ExitOverride>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExitAction {
    #[serde(default = "ExitAction::default_trap")]
    pub trap: bool,

    #[serde(default = "RegSet::all")]
    pub read_set: RegSet,

    #[serde(default = "RegSet::all")]
    pub write_set: RegSet,
}

impl Default for ExitAction {
    fn default() -> Self {
        Self {
            trap: true,
            read_set: RegSet::all(),
            write_set: RegSet::all(),
        }
    }
}

impl ExitAction {
    fn default_trap() -> bool {
        true
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExitOverride {
    /// VMEXIT reason.  Symbolic names (e.g. `"EPT_VIOLATION"`) accepted;
    /// numeric fallback via decimal or `"0x..."` string.
    pub reason: ExitReason,

    #[serde(flatten)]
    pub action: ExitAction,
}

/// Accept both `"EPT_VIOLATION"` (symbolic) and `42` / `"0x2a"` (numeric).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ExitReason {
    Name(String),
    Number(u32),
}

// ── policies.msrs ───────────────────────────────────────────────────────── //

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MsrsConfig {
    #[serde(default = "MsrsConfig::default_default")]
    pub default: DefaultAction,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overrides: Vec<MsrOverride>,
}

impl Default for MsrsConfig {
    fn default() -> Self {
        Self {
            default: MsrsConfig::default_default(),
            overrides: Vec::new(),
        }
    }
}

impl MsrsConfig {
    fn default_default() -> DefaultAction {
        DefaultAction::Native
    }
}

/// Default action for an MSR or CPUID policy.  Note: `Emulate` only makes
/// sense as an override target (needs a value), so it does not appear here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum DefaultAction {
    Native,
    Trap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum OverrideAction {
    Native,
    Trap,
    Emulate,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MsrOverride {
    /// Single-MSR form.  Mutually exclusive with `range`.
    #[serde(default, skip_serializing_if = "Option::is_none", with = "hex_or_dec_u32_opt")]
    pub msr: Option<u32>,

    /// Inclusive `[lo, hi]` range.  Mutually exclusive with `msr`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<[u32; 2]>,

    pub action: OverrideAction,

    /// Stored value for `Emulate` actions.  Accepts decimal, `"0x..."`, or
    /// a small expression like `"0xFEE00000 | EN | EXTD"` (parsed at apply
    /// time; the config layer only preserves the source string via
    /// [`ValueOrExpr::Expr`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<ValueOrExpr>,
}

impl MsrOverride {
    /// True iff the two overrides target overlapping MSR indices.
    fn matches_overlap(&self, other: &MsrOverride) -> bool {
        let a = self.range_tuple();
        let b = other.range_tuple();
        match (a, b) {
            (Some((al, ah)), Some((bl, bh))) => al <= bh && bl <= ah,
            _ => false,
        }
    }

    fn range_tuple(&self) -> Option<(u32, u32)> {
        match (self.msr, self.range) {
            (Some(m), None) => Some((m, m)),
            (None, Some([lo, hi])) => Some((lo, hi)),
            _ => None,
        }
    }
}

// ── policies.cpuid ──────────────────────────────────────────────────────── //

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CpuidConfig {
    #[serde(default = "CpuidConfig::default_default")]
    pub default: DefaultAction,

    /// Shortcut: install `CPUID_RANGE(0x40000000..=0x4FFFFFFF, Native)` for
    /// the hypervisor CPUID range (default true — matches today's behaviour).
    #[serde(default = "CpuidConfig::default_hypervisor_range_native")]
    pub hypervisor_range_native: bool,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overrides: Vec<CpuidOverride>,
}

impl Default for CpuidConfig {
    fn default() -> Self {
        Self {
            default: CpuidConfig::default_default(),
            hypervisor_range_native: CpuidConfig::default_hypervisor_range_native(),
            overrides: Vec::new(),
        }
    }
}

impl CpuidConfig {
    fn default_default() -> DefaultAction {
        DefaultAction::Native
    }
    fn default_hypervisor_range_native() -> bool {
        true
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CpuidOverride {
    /// Single leaf.  Mutually exclusive with `range`.
    #[serde(default, skip_serializing_if = "Option::is_none", with = "hex_or_dec_u32_opt")]
    pub leaf: Option<u32>,

    /// Subleaf for a single-leaf override (defaults to 0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subleaf: Option<u32>,

    /// Inclusive `[[leaf_lo, subleaf_lo], [leaf_hi, subleaf_hi]]` range.
    /// Mutually exclusive with `leaf`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<[CpuidCoord; 2]>,

    pub action: OverrideAction,

    /// Emulate values.  All four accept `"auto:vtom_bit"` sentinel (resolved
    /// at seal time), decimal/hex integers, or 4-char ASCII strings for
    /// register-encoded signatures.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eax: Option<ValueOrExpr>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ebx: Option<ValueOrExpr>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ecx: Option<ValueOrExpr>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edx: Option<ValueOrExpr>,
}

/// `[leaf, subleaf]` pair used inside CPUID range overrides.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum CpuidCoord {
    Pair([ValueOrExpr; 2]),
}

// ── comm.ivshmem ────────────────────────────────────────────────────────── //

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CommConfig {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ivshmem: Vec<IvshmemEntry>,
}

/// Mirror of the Themis-relevant subset of
/// `cloud-hypervisor::vmm::vm_config::IvshmemConfig`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IvshmemEntry {
    /// Stable id (defaults to array index at reconciliation time).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<u32>,

    pub path: PathBuf,

    /// Size in bytes; accepts decimal or `"0x..."` string.  Unit-suffixed
    /// strings ("2MiB") are TODO and will be added in
    /// `chv-themis-config-ivshmem-merge`.
    #[serde(with = "hex_or_dec_u64")]
    pub size: u64,

    /// Capability-backed mode: `"alias"`, `"carve"`, `"plug"`, or `"none"`
    /// (equivalent to a vanilla `--ivshmem` device).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capa_mode: Option<String>,

    /// Number of additional domains that may plug in (creator only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,

    /// CPUID discovery leaf (defaults to `0x40000004`).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "hex_or_dec_u32_opt"
    )]
    pub cpuid_leaf: Option<u32>,
}

// ── Shared value types ──────────────────────────────────────────────────── //

/// Read/write register set on an interrupt or VMEXIT policy.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum RegSet {
    /// `"ALL"` or `"NONE"` literal.
    Symbolic(RegSetSymbolic),
    /// Explicit list of register names (e.g. `["RAX", "RBX"]`).
    List(Vec<String>),
    /// Numeric bitmap (for escape hatches).
    Bitmap(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum RegSetSymbolic {
    ALL,
    NONE,
}

impl RegSet {
    pub fn all() -> Self {
        RegSet::Symbolic(RegSetSymbolic::ALL)
    }
    pub fn none() -> Self {
        RegSet::Symbolic(RegSetSymbolic::NONE)
    }
}

/// A stored value for an Emulate action.  Preserved verbatim from the
/// config file; expression parsing (e.g. `"0xFEE00000 | EN | EXTD"`) and
/// `"auto:*"` sentinel resolution happen at apply time.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ValueOrExpr {
    /// Plain integer literal from JSON.
    Int(u64),
    /// String form: `"0x..."`, decimal, `"auto:..."`, or expression.
    Str(String),
}

// ── serde helpers: hex-or-decimal integers ──────────────────────────────── //

mod hex_or_dec_u64 {
    use serde::{de::Error as _, Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &u64, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(*v)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Any {
            N(u64),
            S(String),
        }
        match Any::deserialize(d)? {
            Any::N(n) => Ok(n),
            Any::S(s) => parse_u64(&s).map_err(D::Error::custom),
        }
    }

    fn parse_u64(s: &str) -> Result<u64, String> {
        let s = s.trim();
        if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
            u64::from_str_radix(hex, 16).map_err(|e| e.to_string())
        } else {
            s.parse::<u64>().map_err(|e| e.to_string())
        }
    }
}

mod hex_or_dec_u64_opt {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Option<u64>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(n) => s.serialize_some(n),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<u64>, D::Error> {
        Ok(Option::<super::HexOrDecU64>::deserialize(d)?.map(|w| w.0))
    }
}

mod hex_or_dec_u32_opt {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &Option<u32>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(n) => s.serialize_some(n),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<u32>, D::Error> {
        Ok(Option::<super::HexOrDecU32>::deserialize(d)?.map(|w| w.0))
    }
}

/// Newtype wrapper so `Option<HexOrDecU64>` can round-trip hex strings.
#[derive(Debug, Clone, Copy)]
struct HexOrDecU64(u64);

impl<'de> Deserialize<'de> for HexOrDecU64 {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        use serde::de::Error as _;
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Any {
            N(u64),
            S(String),
        }
        Ok(match Any::deserialize(d)? {
            Any::N(n) => HexOrDecU64(n),
            Any::S(s) => {
                let t = s.trim();
                let n = if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                    u64::from_str_radix(h, 16).map_err(D::Error::custom)?
                } else {
                    t.parse::<u64>().map_err(D::Error::custom)?
                };
                HexOrDecU64(n)
            }
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct HexOrDecU32(u32);

impl<'de> Deserialize<'de> for HexOrDecU32 {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        use serde::de::Error as _;
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Any {
            N(u32),
            S(String),
        }
        Ok(match Any::deserialize(d)? {
            Any::N(n) => HexOrDecU32(n),
            Any::S(s) => {
                let t = s.trim();
                let n = if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                    u32::from_str_radix(h, 16).map_err(D::Error::custom)?
                } else {
                    t.parse::<u32>().map_err(D::Error::custom)?
                };
                HexOrDecU32(n)
            }
        })
    }
}

// ── Tests ───────────────────────────────────────────────────────────────── //

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_json_produces_defaults() {
        let cfg = ThemisConfig::from_json_str("{}").expect("parse");
        assert_eq!(cfg.general, GeneralConfig::default());
        assert_eq!(cfg.policies, PoliciesConfig::default());
        assert!(cfg.comm.is_none());
        assert!(cfg.extends.is_none());
    }

    #[test]
    fn confidential_flag_and_hex_vtom() {
        let cfg = ThemisConfig::from_json_str(
            r#"{
                "general": { "confidential": true, "vtom_bit": 46 }
            }"#,
        )
        .expect("parse");
        assert!(cfg.general.confidential);
        assert_eq!(cfg.general.vtom_bit, Some(46));
    }

    #[test]
    fn msr_policy_hex_string() {
        let cfg = ThemisConfig::from_json_str(
            r#"{
                "policies": {
                    "msrs": {
                        "default": "Native",
                        "overrides": [
                            { "msr": "0x6E0", "action": "Emulate", "value": 0 },
                            { "range": [2048, 2095], "action": "Native" }
                        ]
                    }
                }
            }"#,
        )
        .expect("parse");
        assert_eq!(cfg.policies.msrs.default, DefaultAction::Native);
        assert_eq!(cfg.policies.msrs.overrides.len(), 2);
        assert_eq!(cfg.policies.msrs.overrides[0].msr, Some(0x6E0));
        assert_eq!(cfg.policies.msrs.overrides[1].range, Some([2048, 2095]));
    }

    #[test]
    fn api_allow_string_and_list() {
        let all = ThemisConfig::from_json_str(r#"{"policies":{"api":{"allow":"ALL"}}}"#).unwrap();
        assert!(matches!(all.policies.api.allow, ApiAllow::All));
        let list = ThemisConfig::from_json_str(
            r#"{"policies":{"api":{"allow":["carve","seal"]}}}"#,
        )
        .unwrap();
        assert!(matches!(list.policies.api.allow, ApiAllow::List(_)));
    }

    #[test]
    fn interrupts_default_and_override() {
        let cfg = ThemisConfig::from_json_str(
            r#"{
                "policies": {
                    "interrupts": {
                        "default": { "visibility": "Report", "read_set": "ALL", "write_set": "ALL" },
                        "overrides": [
                            { "vector": 236, "visibility": "Deliver", "read_set": "NONE", "write_set": "NONE" }
                        ]
                    }
                }
            }"#,
        ).expect("parse");
        assert_eq!(cfg.policies.interrupts.overrides.len(), 1);
        assert_eq!(cfg.policies.interrupts.overrides[0].vector, 236);
        assert_eq!(
            cfg.policies.interrupts.overrides[0].policy.visibility,
            Visibility::Deliver
        );
    }

    #[test]
    fn extends_builtin_standard_loads() {
        // Bare `"extends": "standard"` must resolve without I/O and copy the
        // built-in defaults into the caller.
        let mut cfg = ThemisConfig::from_json_str(r#"{"extends":"standard"}"#).unwrap();
        cfg.resolve_extends(std::path::Path::new("/tmp/child.json"))
            .unwrap();
        assert!(!cfg.general.confidential);
        // The standard profile installs an Emulate(0x6E0) override.
        assert!(cfg
            .policies
            .msrs
            .overrides
            .iter()
            .any(|o| o.msr == Some(0x6E0)));
    }

    #[test]
    fn extends_builtin_confidential_loads() {
        let mut cfg = ThemisConfig::from_json_str(r#"{"extends":"confidential"}"#).unwrap();
        cfg.resolve_extends(std::path::Path::new("/tmp/child.json"))
            .unwrap();
        assert!(cfg.general.confidential);
        // Confidential profile adds the CoCo detection CPUID leaf.
        assert!(cfg
            .policies
            .cpuid
            .overrides
            .iter()
            .any(|o| o.leaf == Some(0x40000100)));
    }

    #[test]
    fn extends_unknown_builtin_rejected_with_named_error() {
        let mut cfg = ThemisConfig::from_json_str(r#"{"extends":"typo"}"#).unwrap();
        let err = cfg
            .resolve_extends(std::path::Path::new("/tmp/child.json"))
            .unwrap_err();
        assert!(matches!(
            err,
            ThemisConfigError::ExtendsUnknownBuiltin { .. }
        ));
    }

    #[test]
    fn builtin_default_direct_load() {
        let s = ThemisConfig::builtin_default(DefaultProfile::Standard).unwrap();
        assert!(!s.general.confidential);
        let c = ThemisConfig::builtin_default(DefaultProfile::Confidential).unwrap();
        assert!(c.general.confidential);
    }

    #[test]
    fn extends_path_shallow_merges() {
        // Base file on disk with a non-default general section.
        let tmp = std::env::temp_dir().join("themis-config-test-base.json");
        std::fs::write(
            &tmp,
            r#"{"general":{"confidential":true,"vtom_bit":42}}"#,
        )
        .unwrap();

        let child_json = format!(
            r#"{{"extends":"{}"}}"#,
            tmp.display().to_string().replace('\\', "\\\\")
        );
        let mut cfg = ThemisConfig::from_json_str(&child_json).unwrap();
        cfg.resolve_extends(std::path::Path::new("/tmp/child.json"))
            .unwrap();

        assert!(cfg.general.confidential);
        assert_eq!(cfg.general.vtom_bit, Some(42));

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn unknown_field_rejected() {
        let err =
            ThemisConfig::from_json_str(r#"{"general":{"typo_field":true}}"#).unwrap_err();
        assert!(matches!(err, ThemisConfigError::Parse { .. }));
    }

    #[test]
    fn ivshmem_entry_roundtrip() {
        let cfg = ThemisConfig::from_json_str(
            r#"{
                "comm": {
                    "ivshmem": [
                        {
                            "path": "/tmp/shm",
                            "size": "0x200000",
                            "capa_mode": "carve",
                            "count": 2,
                            "cpuid_leaf": "0x40000004"
                        }
                    ]
                }
            }"#,
        )
        .expect("parse");
        let ivs = &cfg.comm.as_ref().unwrap().ivshmem;
        assert_eq!(ivs.len(), 1);
        assert_eq!(ivs[0].size, 0x200000);
        assert_eq!(ivs[0].capa_mode.as_deref(), Some("carve"));
        assert_eq!(ivs[0].cpuid_leaf, Some(0x40000004));
    }
}
