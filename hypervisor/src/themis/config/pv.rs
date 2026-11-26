use serde::{Deserialize, Serialize};
use themis_abi::pv::{
    self, Channel, Descriptor, Identifier, Region, Service, descriptor::range_end,
};

use super::{CommConfig, Result, ThemisConfigError, ValueOrExpr};

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PvRegion {
    pub id: String,
    pub kind: RegionKind,
    pub peer: Peer,
    pub gpa: ValueOrExpr,
    #[serde(
        default,
        with = "super::hex_or_dec_u64_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub size: Option<u64>,
    #[serde(
        default,
        with = "super::hex_or_dec_u64_opt",
        skip_serializing_if = "Option::is_none"
    )]
    pub size_per_vp: Option<u64>,
    pub rights: SharedRights,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum RegionKind {
    #[serde(rename = "alias")]
    Alias,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum Peer {
    #[serde(rename = "parent")]
    Parent,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum SharedRights {
    RW,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PvChannel {
    pub id: String,
    pub protocol: Protocol,
    pub region: String,
    pub io_region: String,
    pub notification: Notification,
    pub scheduling: Scheduling,
    pub services: Vec<ServiceName>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum Protocol {
    #[serde(rename = "themis-pv/1")]
    V1,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum Notification {
    #[serde(rename = "doorbell")]
    Doorbell,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum Scheduling {
    #[serde(rename = "sync")]
    Sync,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ServiceName {
    Control,
    Platform,
    Virtio,
}

impl ServiceName {
    fn bit(self) -> u64 {
        match self {
            Self::Control => Service::Control.bit(),
            Self::Platform => Service::Platform.bit(),
            Self::Virtio => Service::Virtio.bit(),
        }
    }
}

pub(crate) struct ResolvedPv {
    pub descriptor_gpa: u64,
    pub descriptor: Descriptor,
}

fn invalid(message: impl Into<String>) -> ThemisConfigError {
    ThemisConfigError::Validation(message.into())
}

fn wire_error(error: pv::Error) -> ThemisConfigError {
    invalid(format!("PV resource layout: {error:?}"))
}

impl CommConfig {
    pub(super) fn validate_pv(&self) -> Result<()> {
        if self.regions.is_empty() && self.channels.is_empty() {
            return Ok(());
        }
        self.resolve_pv(1, &[], 1 << 48, [1, 1])?;
        Ok(())
    }

    pub(crate) fn resolve_pv(
        &self,
        num_vps: u32,
        occupied: &[(u64, u64)],
        limit: u64,
        generation: [u64; 2],
    ) -> Result<ResolvedPv> {
        if self.regions.len() > pv::MAX_REGIONS || self.channels.len() > pv::MAX_CHANNELS {
            return Err(invalid("too many PV regions or channels"));
        }
        for &(start, size) in occupied {
            range_end(start, size, limit).map_err(wire_error)?;
        }
        let mut allocator = Allocator {
            occupied: occupied.to_vec(),
            limit,
        };
        let mut descriptor = Descriptor::new(generation, num_vps);
        for region in &self.regions {
            let size = match (region.size, region.size_per_vp) {
                (Some(size), None) => size,
                (None, Some(size)) => size
                    .checked_mul(u64::from(num_vps))
                    .ok_or_else(|| invalid("PV per-VP size overflow"))?,
                _ => {
                    return Err(invalid(
                        "PV region requires exactly one size or size_per_vp",
                    ));
                }
            };
            descriptor
                .push_region(Region {
                    id: Identifier::new(&region.id).map_err(wire_error)?,
                    gpa: allocator.place(&region.gpa, size)?,
                    size,
                })
                .map_err(wire_error)?;
        }
        for channel in &self.channels {
            descriptor
                .push_channel(self.resolve_channel(channel)?)
                .map_err(wire_error)?;
        }
        let descriptor_gpa =
            allocator.place(&ValueOrExpr::Str("auto".into()), pv::PAGE_SIZE as u64)?;
        descriptor
            .validate_layout(descriptor_gpa, limit)
            .map_err(wire_error)?;
        Ok(ResolvedPv {
            descriptor_gpa,
            descriptor,
        })
    }

    fn resolve_channel(&self, channel: &PvChannel) -> Result<Channel> {
        let region_index = |id: &str| {
            self.regions
                .iter()
                .position(|r| r.id == id)
                .map(|i| i as u32)
                .ok_or_else(|| invalid(format!("PV channel references missing region {id:?}")))
        };
        let mut services = 0;
        for service in &channel.services {
            if services & service.bit() != 0 {
                return Err(invalid("duplicate PV service"));
            }
            services |= service.bit();
        }
        if services == 0 {
            return Err(invalid("PV channel requires services"));
        }
        let control = region_index(&channel.region)?;
        let io = region_index(&channel.io_region)?;
        if self.regions[control as usize].size_per_vp != Some(pv::PAGE_SIZE as u64)
            || self.regions[control as usize].size.is_some()
        {
            return Err(invalid("PV control region requires size_per_vp = 4096"));
        }
        if self.regions[io as usize].size.is_none()
            || self.regions[io as usize].size_per_vp.is_some()
        {
            return Err(invalid("PV I/O region requires a fixed size"));
        }
        Ok(Channel {
            id: Identifier::new(&channel.id).map_err(wire_error)?,
            control,
            io,
            services: services | Service::Control.bit(),
            doorbell_recipient: 0,
            doorbell_port: 0,
        })
    }
}

struct Allocator {
    occupied: Vec<(u64, u64)>,
    limit: u64,
}

impl Allocator {
    fn place(&mut self, placement: &ValueOrExpr, size: u64) -> Result<u64> {
        let explicit = match placement {
            ValueOrExpr::Int(value) => Some(*value),
            ValueOrExpr::Str(value) if value == "auto" => None,
            ValueOrExpr::Str(value) => Some(
                super::HexOrDecU64::deserialize(serde_json::Value::String(value.clone()))
                    .map_err(|e| invalid(format!("invalid PV GPA: {e}")))?
                    .0,
            ),
        };
        let mut candidate = explicit.unwrap_or(pv::AUTO_GPA_START);
        loop {
            let end = range_end(candidate, size, self.limit).map_err(wire_error)?;
            let conflict = self
                .occupied
                .iter()
                .map(|&(start, len)| (start, start + len))
                .find(|&(start, stop)| candidate < stop && start < end);
            match conflict {
                Some(_) if explicit.is_some() => {
                    return Err(invalid("PV GPA overlaps reserved memory"));
                }
                Some((_, stop)) => {
                    candidate = stop
                        .checked_add(pv::PAGE_SIZE as u64 - 1)
                        .map(|v| v & !(pv::PAGE_SIZE as u64 - 1))
                        .ok_or_else(|| invalid("PV placement overflow"))?;
                }
                None => {
                    self.occupied.push((candidate, size));
                    return Ok(candidate);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::themis::ThemisConfig;

    const CONFIG: &str = r#"{"comm":{
      "regions":[
        {"id":"control","kind":"alias","peer":"parent","gpa":"auto",
         "size_per_vp":"0x1000","rights":"RW"},
        {"id":"io","kind":"alias","peer":"parent","gpa":"auto",
         "size":"0x100000","rights":"RW"}],
      "channels":[{"id":"parent","protocol":"themis-pv/1","region":"control",
        "io_region":"io","notification":"doorbell","scheduling":"sync","services":["control"]}]
    }}"#;

    fn config() -> CommConfig {
        ThemisConfig::from_json_str(CONFIG).unwrap().comm.unwrap()
    }

    #[test]
    fn resolves_per_vp_layout_around_existing_mappings() {
        let occupied = [(pv::AUTO_GPA_START, pv::MAX_IO_SIZE)];
        let result = config().resolve_pv(2, &occupied, 1 << 45, [1, 2]).unwrap();
        assert_eq!(result.descriptor.num_vps, 2);
        assert_eq!(
            result.descriptor.regions()[0].gpa,
            pv::AUTO_GPA_START + pv::MAX_IO_SIZE
        );
        assert_eq!(
            result.descriptor.regions()[0].size,
            2 * pv::PAGE_SIZE as u64
        );
        result
            .descriptor
            .validate_layout(result.descriptor_gpa, 1 << 45)
            .unwrap();
    }

    #[test]
    fn rejects_missing_duplicate_or_multiply_owned_regions() {
        let mut cfg = config();
        cfg.channels[0].io_region = "absent".into();
        assert!(cfg.validate_pv().is_err());
        let mut cfg = config();
        cfg.regions[1].id = "control".into();
        assert!(cfg.validate_pv().is_err());
        let mut cfg = config();
        cfg.channels.push(cfg.channels[0].clone());
        assert!(cfg.validate_pv().is_err());
    }

    #[test]
    fn rejects_ambiguous_size_oversized_pool_and_explicit_overlap() {
        let mut cfg = config();
        cfg.regions[0].size = Some(pv::PAGE_SIZE as u64);
        assert!(cfg.validate_pv().is_err());
        let mut cfg = config();
        cfg.regions[1].size = Some(pv::MAX_IO_SIZE + pv::PAGE_SIZE as u64);
        assert!(cfg.validate_pv().is_err());
        let mut cfg = config();
        cfg.regions[1].gpa = ValueOrExpr::Int(pv::AUTO_GPA_START);
        assert!(cfg.validate_pv().is_err());
    }

    #[test]
    fn rejects_unsupported_modes_and_service_names() {
        for (old, new) in [
            ("\"RW\"", "\"RWX\""),
            ("\"alias\"", "\"carve\""),
            ("\"sync\"", "\"async\""),
            ("\"services\":[\"control\"]", "\"services\":[\"unknown\"]"),
        ] {
            ThemisConfig::from_json_str(&CONFIG.replace(old, new)).unwrap_err();
        }
    }

    #[test]
    fn rejects_vp_dependent_size_validation() {
        let mut cfg = config();
        cfg.regions[0].size_per_vp = None;
        cfg.regions[0].size = Some(2 * pv::PAGE_SIZE as u64);
        assert!(cfg.validate_pv().is_err());
        let mut cfg = config();
        cfg.regions[1].size = None;
        cfg.regions[1].size_per_vp = Some(pv::PAGE_SIZE as u64);
        assert!(cfg.validate_pv().is_err());
    }
}
