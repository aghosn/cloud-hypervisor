use std::fs::{File, OpenOptions};
use std::io::Read;
use std::os::fd::{AsRawFd, RawFd};
use std::sync::{Arc, Mutex};

use anyhow::{Context, bail};
use themis_abi::pv::{
    self, Completion, ControlPage, Descriptor, Identity, Request, Server, Service, Status,
};
use vm_memory::bitmap::AtomicBitmap;
use vm_memory::{GuestAddress, GuestMemoryMmap, GuestRegionMmap};
use vmm_sys_util::eventfd::EventFd;

use super::abi::{
    THHV_CREATE_DOORBELL, THHV_DOORBELL_EVENTFD, THHV_SEND_CAP, ThhvCreateDoorbell,
    ThhvDoorbellEventfd, ThhvSendCap, ThhvSetGuestMemory,
};
use super::config::{ResolvedPv, ThemisConfig};
use super::consts::{THHV_MEM_F_ALIAS, THHV_MEM_R_READ, THHV_MEM_R_WRITE};
use super::helpers::ioctl_with_mut_ref;
use super::ioevent::IoEvents;
use super::mmap::MmapRegion;
use crate::vm::VmOps;

pub(super) fn cpuid_apic_ids(
    tables: &[Vec<crate::arch::x86::CpuIdEntry>],
) -> anyhow::Result<Vec<u32>> {
    let ids = tables
        .iter()
        .map(|entries| {
            entries
                .iter()
                .find(|e| e.function == 0xb && e.index == 0)
                .or_else(|| entries.iter().find(|e| e.function == 0x1f && e.index == 0))
                .map(|e| e.edx)
                .or_else(|| {
                    entries
                        .iter()
                        .find(|e| e.function == 1 && e.index == 0)
                        .map(|e| e.ebx >> 24)
                })
                .ok_or_else(|| anyhow::anyhow!("guest VP has no CPUID APIC identity"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let unique: std::collections::BTreeSet<_> = ids.iter().collect();
    anyhow::ensure!(unique.len() == ids.len(), "duplicate guest APIC IDs");
    Ok(ids)
}

pub(super) fn xapic_logical_targets(
    destination: u32,
    lapic: &[[u32; 1024]],
) -> anyhow::Result<Vec<usize>> {
    let mut targets = Vec::new();
    for (vp, registers) in lapic.iter().enumerate() {
        let logical = registers[0xd0 / 4] >> 24;
        let selected = match registers[0xe0 / 4] >> 28 {
            15 => destination & logical != 0,
            0 => destination >> 4 == logical >> 4 && destination & logical & 15 != 0,
            _ => anyhow::bail!("unsupported xAPIC destination format"),
        };
        if selected {
            targets.push(vp);
        }
    }
    anyhow::ensure!(
        !targets.is_empty(),
        "IPI has no matching guest logical APIC"
    );
    Ok(targets)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CpuRequest {
    FixedIpi {
        vector: u8,
        shorthand: u8,
        logical: bool,
        destination: u32,
    },
    TscDeadline {
        deadline: u64,
    },
    TimerConfig(u32),
}

pub(super) fn ipi_targets(
    caller: u32,
    shorthand: u8,
    logical: bool,
    destination: u32,
    apic_ids: &[u32],
) -> anyhow::Result<Vec<usize>> {
    anyhow::ensure!((caller as usize) < apic_ids.len(), "invalid IPI caller VP");
    let targets = apic_ids
        .iter()
        .enumerate()
        .filter_map(|(vp, &apic_id)| {
            let selected = match shorthand {
                0 if destination == u32::MAX => true,
                0 if logical => {
                    apic_id >> 4 == destination >> 16 && destination & (1 << (apic_id & 15)) != 0
                }
                0 => apic_id == destination,
                1 => vp == caller as usize,
                2 => true,
                3 => vp != caller as usize,
                _ => false,
            };
            selected.then_some(vp)
        })
        .collect::<Vec<_>>();
    anyhow::ensure!(
        !targets.is_empty() || shorthand == 3,
        "IPI has no matching guest VP"
    );
    Ok(targets)
}

pub(super) struct PvResources {
    pub descriptor_gpa: u64,
    pub descriptor: Descriptor,
    regions: Vec<Arc<GuestRegionMmap<AtomicBitmap>>>,
    _descriptor_backing: MmapRegion,
    servers: Vec<Mutex<Server>>,
    notifications: Vec<EventFd>,
    doorbell_owner: Option<File>,
    platform_lock: Mutex<()>,
}

impl PvResources {
    pub fn prepare(
        config: &ThemisConfig,
        num_vps: u32,
        occupied: &[(u64, u64)],
        limit: u64,
    ) -> anyhow::Result<Option<Self>> {
        let Some(comm) = &config.comm else {
            if config.general.device_transport == Some(super::config::DeviceTransport::Pv) {
                bail!("PV device transport requires a declared Control+Platform channel");
            }
            return Ok(None);
        };
        if comm.regions.is_empty() && comm.channels.is_empty() {
            if config.general.device_transport == Some(super::config::DeviceTransport::Pv) {
                bail!("PV device transport requires a declared Control+Platform channel");
            }
            return Ok(None);
        }
        let mut entropy = [0; 16];
        File::open("/dev/urandom")?.read_exact(&mut entropy)?;
        let generation = [
            u64::from_le_bytes(entropy[..8].try_into()?),
            u64::from_le_bytes(entropy[8..].try_into()?),
        ];
        let ResolvedPv {
            descriptor_gpa,
            mut descriptor,
        } = comm.resolve_pv(num_vps, occupied, limit, generation)?;
        descriptor.device_io = config.general.uses_pv_devices();
        descriptor
            .validate_layout(descriptor_gpa, limit)
            .map_err(|error| anyhow::anyhow!("PV device transport layout: {error:?}"))?;
        if descriptor
            .channels()
            .iter()
            .any(|c| c.services & !(Service::Control.bit() | Service::Platform.bit()) != 0)
        {
            bail!("PV virtio service is not implemented yet");
        }
        let regions = descriptor
            .regions()
            .iter()
            .map(|r| {
                let mapping = vm_memory::MmapRegion::build(
                    None,
                    r.size as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_ANONYMOUS,
                )?;
                Ok(Arc::new(GuestRegionMmap::new(
                    mapping,
                    GuestAddress(r.gpa),
                )?))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let descriptor_backing = MmapRegion::new_shared_anonymous(pv::PAGE_SIZE)?;
        let mut servers = Vec::new();
        for (index, channel) in descriptor.channels().iter().enumerate() {
            for vp in 0..num_vps {
                let offset = vp as usize * pv::PAGE_SIZE;
                let mapping = &regions[channel.control as usize];
                // SAFETY: validated control size is exactly num_vps aligned pages.
                unsafe {
                    mapping
                        .as_ptr()
                        .cast::<u8>()
                        .add(offset)
                        .cast::<ControlPage>()
                        .write(ControlPage::new());
                }
                servers.push(Mutex::new(Server::new(Identity {
                    generation,
                    channel: index as u32,
                    vp,
                })));
            }
        }
        Ok(Some(Self {
            descriptor_gpa,
            descriptor,
            regions,
            _descriptor_backing: descriptor_backing,
            servers,
            notifications: Vec::new(),
            doorbell_owner: None,
            platform_lock: Mutex::new(()),
        }))
    }

    pub fn memory_regions(&self, shared_mask: u64) -> Vec<ThhvSetGuestMemory> {
        let mut result = self
            .descriptor
            .regions()
            .iter()
            .zip(&self.regions)
            .map(|(region, mapping)| ThhvSetGuestMemory {
                guest_pfn: region.gpa >> 12,
                userspace_addr: mapping.as_ptr() as u64,
                size: region.size,
                flags: THHV_MEM_F_ALIAS,
                rights: THHV_MEM_R_READ | THHV_MEM_R_WRITE,
                ..ThhvSetGuestMemory::default()
            })
            .collect::<Vec<_>>();
        if shared_mask != 0 {
            result.extend(result.clone().into_iter().map(|mut region| {
                region.guest_pfn |= shared_mask >> 12;
                region
            }));
        }
        result.push(ThhvSetGuestMemory {
            guest_pfn: self.descriptor_gpa >> 12,
            userspace_addr: self._descriptor_backing.as_u64(),
            size: pv::PAGE_SIZE as u64,
            rights: THHV_MEM_R_READ,
            ..ThhvSetGuestMemory::default()
        });
        result
    }

    pub fn device_memory(&self) -> anyhow::Result<GuestMemoryMmap<AtomicBitmap>> {
        let mut regions: Vec<_> = self
            .descriptor
            .channels()
            .iter()
            .map(|channel| self.regions[channel.io as usize].clone())
            .collect();
        regions.sort_by_key(|region| vm_memory::GuestMemoryRegion::start_addr(region.as_ref()));
        GuestMemoryMmap::from_arc_regions(regions).context("PV device IO memory view")
    }

    pub fn register_doorbells(&mut self, fd: RawFd, limit: u64) -> anyhow::Result<()> {
        let owner = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/thhv")?;
        for index in 0..self.descriptor.channels().len() {
            let event = EventFd::new(libc::EFD_CLOEXEC | libc::EFD_NONBLOCK)?;
            let mut doorbell = ThhvCreateDoorbell::default(); // Sync, no stored VP.
            ioctl_with_mut_ref(owner.as_raw_fd(), THHV_CREATE_DOORBELL, &mut doorbell)
                .context("create PV notification capability")?;
            let mut subscription = ThhvDoorbellEventfd {
                port: doorbell.port,
                fd: event.as_raw_fd(),
                reserved: 0,
            };
            ioctl_with_mut_ref(owner.as_raw_fd(), THHV_DOORBELL_EVENTFD, &mut subscription)
                .context("subscribe to PV recipient port")?;
            let mut send = ThhvSendCap {
                handle: doorbell.handle,
                attrs: 0,
            };
            ioctl_with_mut_ref(fd, THHV_SEND_CAP, &mut send)
                .context("send PV notification capability before seal")?;
            self.descriptor
                .bind_doorbell(index, doorbell.recipient, doorbell.port)
                .map_err(|e| anyhow::anyhow!("bind PV notification identity: {e:?}"))?;
            self.notifications.push(event);
        }
        self.descriptor
            .validate(self.descriptor_gpa, limit)
            .map_err(|e| anyhow::anyhow!("final PV descriptor: {e:?}"))?;
        let mut bytes = [0; pv::PAGE_SIZE];
        self.descriptor.encode(&mut bytes);
        // SAFETY: owned page-sized mapping, finalized before sending it to the child.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self._descriptor_backing.addr.cast(),
                bytes.len(),
            );
        }
        self.doorbell_owner = Some(owner);
        Ok(())
    }

    pub fn service_vp(
        &self,
        vp: u32,
        vm_ops: Option<&dyn VmOps>,
        ioevents: &Mutex<IoEvents>,
        cpu_ops: Option<&dyn Fn(CpuRequest) -> anyhow::Result<()>>,
    ) -> anyhow::Result<()> {
        if vp >= self.descriptor.num_vps {
            bail!("PV VP index outside resolved launch");
        }
        for (index, channel) in self.descriptor.channels().iter().enumerate() {
            let mapping = &self.regions[channel.control as usize];
            // SAFETY: owned, live mmap, aligned and initialized before guest entry.
            // ControlPage contains only atomic words; no mutable references exist.
            let page = unsafe {
                &*mapping
                    .as_ptr()
                    .cast::<u8>()
                    .add(vp as usize * pv::PAGE_SIZE)
                    .cast::<ControlPage>()
            };
            let server = &self.servers[index * self.descriptor.num_vps as usize + vp as usize];
            server
                .lock()
                .unwrap()
                .service(page, |request| {
                    if channel.services & request.service.bit() == 0 {
                        return Completion::unsupported();
                    }
                    if request.service == Service::Platform {
                        let _guard = self.platform_lock.lock().unwrap();
                        dispatch_platform(request, vm_ops, ioevents, cpu_ops)
                    } else {
                        dispatch_control(request, channel.services)
                    }
                })
                .map_err(|e| anyhow::anyhow!("PV channel {index}, VP {vp}: {e:?}"))?;
        }
        Ok(())
    }
}

fn dispatch_control(request: Request, services: u64) -> Completion {
    if request.service != Service::Control || request.device != 0 {
        return Completion::unsupported();
    }
    let values = match request.opcode {
        pv::OP_NEGOTIATE => [u64::from(pv::VERSION), services, 0, 0, 0, 0],
        pv::OP_ECHO => request.args,
        _ => return Completion::unsupported(),
    };
    Completion {
        status: Status::Success,
        values,
    }
}

fn dispatch_platform(
    request: Request,
    vm_ops: Option<&dyn VmOps>,
    ioevents: &Mutex<IoEvents>,
    cpu_ops: Option<&dyn Fn(CpuRequest) -> anyhow::Result<()>>,
) -> Completion {
    use pv::platform::*;
    let invalid = Completion {
        status: Status::Invalid,
        values: [0; 6],
    };
    if request.device != 0 || request.args[3..].iter().any(|&v| v != 0) {
        return invalid;
    }
    let register = request.args[0];
    let value = request.args[1];
    let data = request.args[2];
    let mut cpu_request = None;
    if !matches!(
        request.opcode,
        PIO_READ | PIO_WRITE | MMIO_READ | MMIO_WRITE
    ) && data != 0
    {
        return invalid;
    }
    match request.opcode {
        IPI_WRITE => {
            let low = register as u32;
            // Fixed, edge-triggered IPI; level-assert is architecturally ignored.
            if value != 0 || low & !0x000c_48ff != 0 || low & 0xff < 16 {
                return invalid;
            }
            cpu_request = Some(CpuRequest::FixedIpi {
                vector: low as u8,
                shorthand: ((low >> 18) & 3) as u8,
                logical: low & (1 << 11) != 0,
                destination: (register >> 32) as u32,
            });
        }
        TSC_DEADLINE => {
            if value != 0 {
                return invalid;
            }
            cpu_request = Some(CpuRequest::TscDeadline { deadline: register });
        }
        TIMER_CONFIG => {
            if !super::timer::valid_lvt(register) || value != 0 {
                return invalid;
            }
            cpu_request = Some(CpuRequest::TimerConfig(register as u32));
        }
        IOAPIC_READ | IOAPIC_WRITE => {
            if !(register <= 2 || (0x10..=0x3f).contains(&register))
                || value > u64::from(u32::MAX)
                || (request.opcode == IOAPIC_READ && value != 0)
                || (request.opcode == IOAPIC_WRITE && register != 0 && register < 0x10)
            {
                return invalid;
            }
        }
        IOAPIC_EOI if register <= 255 && value == 0 => {}
        IOAPIC_SYNC if register == 0 && value == 0 => {}
        IOAPIC_EOI | IOAPIC_SYNC => return invalid,
        PIO_READ | PIO_WRITE => {
            if !matches!(value, 1 | 2 | 4)
                || register > 0x1_0000 - value
                || data >= (1u64 << (value * 8))
                || (request.opcode == PIO_READ && data != 0)
            {
                return invalid;
            }
        }
        MMIO_READ | MMIO_WRITE => {
            if !matches!(value, 1 | 2 | 4 | 8)
                || register.checked_add(value).is_none()
                || (value != 8 && data >= (1u64 << (value * 8)))
                || (request.opcode == MMIO_READ && data != 0)
            {
                return invalid;
            }
        }
        _ => return Completion::unsupported(),
    }
    let operation = || -> anyhow::Result<[u64; 6]> {
        if let Some(request) = cpu_request {
            cpu_ops.ok_or_else(|| anyhow::anyhow!("PV platform has no owning-VP CPU backend"))?(
                request,
            )?;
            return Ok([0; 6]);
        }
        let vm = vm_ops.ok_or_else(|| anyhow::anyhow!("PV platform has no device backend"))?;
        let mut result = [0; 6];
        match request.opcode {
            IOAPIC_READ | IOAPIC_WRITE => {
                vm.device_mmio_write(IOAPIC_BASE, &(register as u32).to_le_bytes())?;
                if request.opcode == IOAPIC_READ {
                    let mut data = [0; 4];
                    vm.device_mmio_read(IOAPIC_BASE + 0x10, &mut data)?;
                    result[0] = u64::from(u32::from_le_bytes(data));
                } else {
                    vm.device_mmio_write(IOAPIC_BASE + 0x10, &(value as u32).to_le_bytes())?;
                }
            }
            IOAPIC_EOI => {
                vm.device_mmio_write(IOAPIC_BASE + 0x40, &(register as u32).to_le_bytes())?;
            }
            IOAPIC_SYNC => vm.device_mmio_read(IOAPIC_BASE + 0x10, &mut [0; 4])?,
            PIO_READ | MMIO_READ => {
                let mut bytes = [0; 8];
                if request.opcode == PIO_READ {
                    vm.pio_read(register, &mut bytes[..value as usize])?;
                } else {
                    vm.device_mmio_read(register, &mut bytes[..value as usize])?;
                }
                result[0] = u64::from_le_bytes(bytes);
            }
            PIO_WRITE | MMIO_WRITE => {
                let bytes = data.to_le_bytes();
                let bytes = &bytes[..value as usize];
                let address = if request.opcode == PIO_WRITE {
                    crate::IoEventAddress::Pio(register)
                } else {
                    crate::IoEventAddress::Mmio(register)
                };
                if !ioevents.lock().unwrap().write(address, bytes)? {
                    if request.opcode == PIO_WRITE {
                        vm.pio_write(register, bytes)?;
                    } else {
                        vm.device_mmio_write(register, bytes)?;
                    }
                }
            }
            _ => unreachable!(),
        }
        Ok(result)
    };
    match operation() {
        Ok(values) => Completion {
            status: Status::Success,
            values,
        },
        Err(error) => {
            eprintln!("PV platform operation {} failed: {error:#}", request.opcode);
            Completion {
                status: Status::Failed,
                values: [0; 6],
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn legacy_ipi_destinations_use_cpuid_identity_and_logical_registers() {
        use crate::arch::x86::CpuIdEntry;
        let tables = [0, 1, 2, 4, 5, 6].map(|id| {
            vec![CpuIdEntry {
                function: 0xb,
                edx: id,
                ..Default::default()
            }]
        });
        let ids = super::cpuid_apic_ids(&tables).unwrap();
        assert_eq!(super::ipi_targets(0, 0, false, 4, &ids).unwrap(), vec![3]);
        assert_eq!(super::ipi_targets(0, 0, false, 6, &ids).unwrap(), vec![5]);
        let mut lapic = [[0u32; 1024]; 2];
        lapic[0][0xd0 / 4] = 1 << 24;
        lapic[1][0xd0 / 4] = 4 << 24;
        for registers in &mut lapic {
            registers[0xe0 / 4] = u32::MAX;
        }
        assert_eq!(super::xapic_logical_targets(4, &lapic).unwrap(), vec![1]);
        lapic[0][0xe0 / 4] = 0x0fff_ffff;
        lapic[1][0xe0 / 4] = 0x0fff_ffff;
        lapic[0][0xd0 / 4] = 0x11 << 24;
        lapic[1][0xd0 / 4] = 0x24 << 24;
        assert_eq!(super::xapic_logical_targets(0x24, &lapic).unwrap(), vec![1]);
        super::xapic_logical_targets(0x14, &lapic).unwrap_err();
    }

    use themis_abi::pv::Client;

    use super::*;
    use crate::vm::{HypervisorVmError, Result as VmResult};

    fn dispatch_platform(request: Request, vm_ops: Option<&dyn VmOps>) -> Completion {
        super::dispatch_platform(request, vm_ops, &Mutex::default(), None)
    }

    #[test]
    fn shared_cpu_requests_validate_operands_and_surface_backend_failure() {
        use std::cell::RefCell;

        use pv::platform::{IPI_WRITE, TIMER_CONFIG, TSC_DEADLINE};

        let calls = RefCell::new(Vec::new());
        let backend = |request| {
            calls.borrow_mut().push(request);
            Ok(())
        };
        let events = Mutex::default();
        let ipi = (0x1234_u64 << 32) | 0x48f0;
        for request in [
            platform_request(IPI_WRITE, ipi, 0),
            platform_request(TSC_DEADLINE, u64::MAX, 0),
            platform_request(TSC_DEADLINE, 0, 0),
            platform_request(TIMER_CONFIG, 0x5_0061, 0),
        ] {
            let reply = super::dispatch_platform(request, None, &events, Some(&backend));
            assert_eq!(reply.status, Status::Success);
            assert_eq!(reply.values, [0; 6]);
        }
        assert_eq!(
            *calls.borrow(),
            [
                CpuRequest::FixedIpi {
                    vector: 0xf0,
                    shorthand: 0,
                    logical: true,
                    destination: 0x1234,
                },
                CpuRequest::TscDeadline { deadline: u64::MAX },
                CpuRequest::TscDeadline { deadline: 0 },
                CpuRequest::TimerConfig(0x5_0061),
            ]
        );
        for request in [
            platform_request(IPI_WRITE, 15, 0),
            platform_request(IPI_WRITE, 0x4f0, 0), // NMI is not fixed delivery.
            platform_request(IPI_WRITE, 0x80f0, 0), // Level-triggered.
            platform_request(IPI_WRITE, 0x8000_00f0, 0),
            platform_request(IPI_WRITE, ipi, 1),
            platform_request(TSC_DEADLINE, 1, 1),
            platform_request(TSC_DEADLINE, 1, 1 << 32),
            platform_request(TIMER_CONFIG, 0x8_0051, 0),
            {
                let mut request = platform_request(TSC_DEADLINE, 1, 0);
                request.args[2] = 1;
                request
            },
            {
                let mut request = platform_request(IPI_WRITE, ipi, 0);
                request.args[5] = 1;
                request
            },
        ] {
            assert_eq!(
                super::dispatch_platform(request, None, &events, Some(&backend)).status,
                Status::Invalid
            );
        }
        assert_eq!(calls.borrow().len(), 4);
        assert_eq!(
            super::dispatch_platform(platform_request(9, 1, 0), None, &events, Some(&backend))
                .status,
            Status::Unsupported,
        );
        let request = platform_request(TSC_DEADLINE, 1, 0);
        assert_eq!(
            super::dispatch_platform(request, None, &events, None).status,
            Status::Failed
        );
        assert_eq!(
            super::dispatch_platform(
                request,
                None,
                &events,
                Some(&|_| { anyhow::bail!("timer programming failed") })
            )
            .status,
            Status::Failed
        );
    }

    #[test]
    fn shared_ipis_resolve_topology_logical_clusters_and_shorthands() {
        let ids = [0, 1, 16, 17, 0x1234];
        assert_eq!(ipi_targets(1, 0, false, 0x1234, &ids).unwrap(), [4]);
        assert_eq!(ipi_targets(1, 0, true, 0x10003, &ids).unwrap(), [2, 3]);
        assert_eq!(ipi_targets(1, 1, false, 0, &ids).unwrap(), [1]);
        assert_eq!(ipi_targets(1, 2, false, 0, &ids).unwrap(), [0, 1, 2, 3, 4]);
        assert_eq!(ipi_targets(1, 3, false, 0, &ids).unwrap(), [0, 2, 3, 4]);
        assert_eq!(
            ipi_targets(1, 0, false, u32::MAX, &ids).unwrap(),
            [0, 1, 2, 3, 4]
        );
        assert!(ipi_targets(0, 3, false, 0, &[0]).unwrap().is_empty());
        ipi_targets(1, 0, false, 99, &ids).unwrap_err();
        ipi_targets(1, 0, true, 0x10000, &ids).unwrap_err();
        ipi_targets(9, 1, false, 0, &ids).unwrap_err();
        ipi_targets(1, 4, false, 0, &ids).unwrap_err();
    }

    #[derive(Default)]
    struct PlatformBackend {
        accesses: Mutex<Vec<(bool, u64, u32)>>,
        ports: Mutex<Vec<(bool, u64, Vec<u8>)>>,
        device: Mutex<Vec<(bool, u64, Vec<u8>)>>,
        selected: std::sync::atomic::AtomicU32,
        fail: bool,
    }

    impl VmOps for PlatformBackend {
        fn guest_mem_write(&self, _: u64, _: &[u8]) -> VmResult<usize> {
            panic!("PV platform must not access private memory");
        }
        fn guest_mem_read(&self, _: u64, _: &mut [u8]) -> VmResult<usize> {
            panic!("PV platform must not access private memory");
        }
        fn mmio_read(&self, gpa: u64, data: &mut [u8]) -> VmResult<()> {
            if self.fail {
                return Err(HypervisorVmError::MmioBusRead(anyhow::anyhow!(
                    "test failure"
                )));
            }
            if gpa == 0x1000 {
                data.copy_from_slice(&0x1234_5678_8765_4321u64.to_le_bytes()[..data.len()]);
                self.device
                    .lock()
                    .unwrap()
                    .push((false, gpa, data.to_vec()));
                return Ok(());
            }
            assert_eq!(gpa, pv::platform::IOAPIC_BASE + 0x10);
            assert_eq!(data.len(), 4);
            let selected = self.selected.load(std::sync::atomic::Ordering::Relaxed);
            data.copy_from_slice(&selected.to_le_bytes());
            self.accesses.lock().unwrap().push((false, gpa, selected));
            Ok(())
        }
        fn mmio_write(&self, gpa: u64, data: &[u8]) -> VmResult<()> {
            if self.fail {
                return Err(HypervisorVmError::MmioBusWrite(anyhow::anyhow!(
                    "test failure"
                )));
            }
            if gpa == 0x1000 {
                self.device.lock().unwrap().push((true, gpa, data.to_vec()));
                return Ok(());
            }
            assert_eq!(data.len(), 4);
            let value = u32::from_le_bytes(data.try_into().unwrap());
            self.accesses.lock().unwrap().push((true, gpa, value));
            if gpa == pv::platform::IOAPIC_BASE {
                self.selected
                    .store(value, std::sync::atomic::Ordering::Relaxed);
                std::thread::yield_now();
            }
            Ok(())
        }
        fn pio_read(&self, port: u64, data: &mut [u8]) -> VmResult<()> {
            if self.fail {
                return Err(HypervisorVmError::IoBusRead(anyhow::anyhow!(
                    "test failure"
                )));
            }
            data.copy_from_slice(&0x8765_4321u32.to_le_bytes()[..data.len()]);
            self.ports
                .lock()
                .unwrap()
                .push((false, port, data.to_vec()));
            Ok(())
        }
        fn pio_write(&self, port: u64, data: &[u8]) -> VmResult<()> {
            if self.fail {
                return Err(HypervisorVmError::IoBusWrite(anyhow::anyhow!(
                    "test failure"
                )));
            }
            self.ports.lock().unwrap().push((true, port, data.to_vec()));
            Ok(())
        }
    }

    fn platform_config() -> ThemisConfig {
        let mut config: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../eunomia/policies/pv/standard.json"
        ))
        .unwrap();
        config["comm"]["channels"][0]["services"] = serde_json::json!(["control", "platform"]);
        ThemisConfig::from_json_str(&config.to_string()).unwrap()
    }

    #[test]
    fn device_transport_selection_is_bound_into_the_bootstrap_descriptor() {
        use super::super::config::DeviceTransport;
        for (confidential, transport, expected) in [
            (false, None, false),
            (false, Some(DeviceTransport::Legacy), false),
            (false, Some(DeviceTransport::Pv), true),
            (true, None, true),
            (true, Some(DeviceTransport::Pv), true),
        ] {
            let mut config = platform_config();
            config.general.confidential = confidential;
            config.general.device_transport = transport;
            let resources = PvResources::prepare(&config, 1, &[], 1 << 45)
                .unwrap()
                .unwrap();
            assert_eq!(resources.descriptor.device_io, expected);
        }
        let mut config = platform_config();
        config.general.device_transport = Some(DeviceTransport::Pv);
        config.comm = None;
        assert!(PvResources::prepare(&config, 1, &[], 1 << 45).is_err());
        config = ThemisConfig::from_json_str(include_str!(
            "../../../../eunomia/policies/pv/standard.json"
        ))
        .unwrap();
        config.general.device_transport = Some(DeviceTransport::Pv);
        assert!(PvResources::prepare(&config, 1, &[], 1 << 45).is_err());
    }

    fn platform_request(opcode: u32, arg0: u64, arg1: u64) -> Request {
        Request {
            service: Service::Platform,
            opcode,
            device: 0,
            args: [arg0, arg1, 0, 0, 0, 0],
        }
    }

    #[test]
    fn platform_dispatch_uses_only_validated_device_transactions() {
        use pv::platform::*;
        let backend = PlatformBackend::default();
        let reply = dispatch_platform(platform_request(IOAPIC_READ, 0x10, 0), Some(&backend));
        assert_eq!(reply.status, Status::Success);
        assert_eq!(reply.values, [0x10, 0, 0, 0, 0, 0]);
        for request in [
            platform_request(IOAPIC_WRITE, 0x11, 0x1234),
            platform_request(IOAPIC_EOI, 0xf0, 0),
            platform_request(IOAPIC_SYNC, 0, 0),
        ] {
            assert_eq!(
                dispatch_platform(request, Some(&backend)).status,
                Status::Success
            );
        }
        assert_eq!(
            *backend.accesses.lock().unwrap(),
            [
                (true, IOAPIC_BASE, 0x10),
                (false, IOAPIC_BASE + 0x10, 0x10),
                (true, IOAPIC_BASE, 0x11),
                (true, IOAPIC_BASE + 0x10, 0x1234),
                (true, IOAPIC_BASE + 0x40, 0xf0),
                (false, IOAPIC_BASE + 0x10, 0x11),
            ]
        );
    }

    #[test]
    fn invalid_platform_requests_never_touch_the_backend() {
        use pv::platform::*;
        let backend = PlatformBackend::default();
        let mut device = platform_request(IOAPIC_READ, 0, 0);
        device.device = 1;
        let mut reserved = platform_request(IOAPIC_READ, 0, 0);
        reserved.args[5] = 1;
        for request in [
            device,
            reserved,
            platform_request(IOAPIC_READ, 3, 0),
            platform_request(IOAPIC_READ, 0x40, 0),
            platform_request(IOAPIC_READ, u64::MAX, 0),
            platform_request(IOAPIC_READ, 1, 1),
            platform_request(IOAPIC_WRITE, 1, 0),
            platform_request(IOAPIC_WRITE, 2, 0),
            platform_request(IOAPIC_WRITE, 0x10, 1 << 32),
            platform_request(IOAPIC_EOI, 256, 0),
            platform_request(IOAPIC_EOI, 255, 1),
            platform_request(IOAPIC_SYNC, 1, 0),
            platform_request(IOAPIC_SYNC, 0, 1),
        ] {
            assert_eq!(
                dispatch_platform(request, Some(&backend)).status,
                Status::Invalid
            );
        }
        assert_eq!(
            dispatch_platform(platform_request(u32::MAX, 0, 0), Some(&backend)).status,
            Status::Unsupported
        );
        assert!(backend.accesses.lock().unwrap().is_empty());
        assert!(backend.ports.lock().unwrap().is_empty());
        let request = platform_request(IOAPIC_READ, 1, 0);
        assert_eq!(dispatch_platform(request, None).status, Status::Failed);
        let failing = PlatformBackend {
            fail: true,
            ..Default::default()
        };
        assert_eq!(
            dispatch_platform(request, Some(&failing)).status,
            Status::Failed
        );
    }

    #[test]
    fn platform_pio_uses_exact_width_shared_operands() {
        use pv::platform::*;
        let backend = PlatformBackend::default();
        for width in [1, 2, 4] {
            let port = 0x1_0000 - width;
            let expected = 0x8765_4321 & ((1u64 << (8 * width)) - 1);
            let request = platform_request(PIO_READ, port, width);
            let reply = dispatch_platform(request, Some(&backend));
            assert_eq!(reply.status, Status::Success);
            assert_eq!(reply.values, [expected, 0, 0, 0, 0, 0]);
            let mut request = platform_request(PIO_WRITE, port, width);
            request.args[2] = expected;
            assert_eq!(
                dispatch_platform(request, Some(&backend)).status,
                Status::Success
            );
            let expected_bytes = expected.to_le_bytes()[..width as usize].to_vec();
            let ports = backend.ports.lock().unwrap();
            assert_eq!(
                ports[ports.len() - 2],
                (false, port, expected_bytes.clone())
            );
            assert_eq!(ports[ports.len() - 1], (true, port, expected_bytes));
        }
        assert!(backend.accesses.lock().unwrap().is_empty());
    }

    #[test]
    fn platform_pio_rejects_malformed_operands_and_reports_backend_failure() {
        use pv::platform::*;
        let backend = PlatformBackend::default();
        let mut read_value = platform_request(PIO_READ, 0x3f8, 1);
        read_value.args[2] = 1;
        let mut reserved = platform_request(PIO_WRITE, 0x3f8, 1);
        reserved.args[3] = 1;
        let mut device = platform_request(PIO_READ, 0x3f8, 1);
        device.device = 1;
        let mut invalid = vec![read_value, reserved, device];
        for width in [0, 3, 8, u64::MAX] {
            invalid.push(platform_request(PIO_READ, 0, width));
        }

        for width in [1, 2, 4] {
            invalid.push(platform_request(PIO_READ, 0x1_0001 - width, width));
            let mut request = platform_request(PIO_WRITE, 0x3f8, width);
            request.args[2] = 1 << (width * 8);
            invalid.push(request);
        }
        invalid.push(platform_request(PIO_READ, u64::MAX, 1));
        for request in invalid {
            assert_eq!(
                dispatch_platform(request, Some(&backend)).status,
                Status::Invalid
            );
        }
        assert!(backend.accesses.lock().unwrap().is_empty());
        assert!(backend.ports.lock().unwrap().is_empty());
        let failing = PlatformBackend {
            fail: true,
            ..Default::default()
        };
        for opcode in [PIO_READ, PIO_WRITE] {
            let request = platform_request(opcode, 0x3f8, 1);
            assert_eq!(dispatch_platform(request, None).status, Status::Failed);
            assert_eq!(
                dispatch_platform(request, Some(&failing)).status,
                Status::Failed
            );
        }
    }

    #[test]
    fn shared_device_notifications_reuse_ioeventfd_matching() {
        for (opcode, address, port) in [
            (
                pv::platform::PIO_WRITE,
                crate::IoEventAddress::Pio(0xc000),
                0xc000,
            ),
            (
                pv::platform::MMIO_WRITE,
                crate::IoEventAddress::Mmio(0x1000),
                0x1000,
            ),
        ] {
            let event = EventFd::new(libc::EFD_NONBLOCK).unwrap();
            let events = Mutex::new(IoEvents::default());
            events
                .lock()
                .unwrap()
                .register(&event, address, Some(crate::vm::DataMatch::DataMatch32(7)))
                .unwrap();
            let backend = PlatformBackend::default();
            let mut request = platform_request(opcode, port, 4);
            request.args[2] = 7;
            assert_eq!(
                super::dispatch_platform(request, Some(&backend), &events, None).status,
                Status::Success
            );
            assert_eq!(event.read().unwrap(), 1);
            assert!(backend.ports.lock().unwrap().is_empty());
            assert!(backend.device.lock().unwrap().is_empty());
            request.args[2] = 8;
            assert_eq!(
                super::dispatch_platform(request, Some(&backend), &events, None).status,
                Status::Success
            );
            let accesses = if opcode == pv::platform::PIO_WRITE {
                backend.ports.lock().unwrap()
            } else {
                backend.device.lock().unwrap()
            };
            assert_eq!(*accesses, [(true, port, 8u32.to_le_bytes().to_vec())]);
        }
    }

    #[test]
    fn shared_mmio_uses_device_bus_only_and_rejects_malformed_operands() {
        use pv::platform::*;
        let backend = PlatformBackend::default();
        for width in [1, 2, 4, 8] {
            let mask = u64::MAX >> (64 - width * 8);
            let expected = 0x1234_5678_8765_4321 & mask;
            let request = platform_request(MMIO_READ, 0x1000, width);
            let reply = dispatch_platform(request, Some(&backend));
            assert_eq!(reply.status, Status::Success);
            assert_eq!(reply.values, [expected, 0, 0, 0, 0, 0]);
            let mut request = platform_request(MMIO_WRITE, 0x1000, width);
            request.args[2] = expected;
            assert_eq!(
                dispatch_platform(request, Some(&backend)).status,
                Status::Success
            );
            let accesses = backend.device.lock().unwrap();
            let bytes = expected.to_le_bytes()[..width as usize].to_vec();
            assert_eq!(accesses[accesses.len() - 2], (false, 0x1000, bytes.clone()));
            assert_eq!(accesses[accesses.len() - 1], (true, 0x1000, bytes));
        }
        let mut invalid = Vec::new();
        for width in [0, 3, 5, 16, u64::MAX] {
            invalid.push(platform_request(MMIO_READ, 0x1000, width));
        }
        for width in [1, 2, 4, 8] {
            invalid.push(platform_request(MMIO_READ, u64::MAX - width + 1, width));
        }
        let mut value = platform_request(MMIO_READ, 0x1000, 8);
        value.args[2] = 1;
        invalid.push(value);
        let mut narrow = platform_request(MMIO_WRITE, 0x1000, 4);
        narrow.args[2] = 1 << 32;
        invalid.push(narrow);
        let mut reserved = platform_request(MMIO_WRITE, 0x1000, 8);
        reserved.args[5] = 1;
        invalid.push(reserved);
        let count = backend.device.lock().unwrap().len();
        for request in invalid {
            assert_eq!(
                dispatch_platform(request, Some(&backend)).status,
                Status::Invalid
            );
        }
        assert_eq!(backend.device.lock().unwrap().len(), count);
        assert!(backend.ports.lock().unwrap().is_empty());
        assert!(backend.accesses.lock().unwrap().is_empty());
        let failing = PlatformBackend {
            fail: true,
            ..Default::default()
        };
        for opcode in [MMIO_READ, MMIO_WRITE] {
            let request = platform_request(opcode, 0x1000, 8);
            assert_eq!(dispatch_platform(request, None).status, Status::Failed);
            assert_eq!(
                dispatch_platform(request, Some(&failing)).status,
                Status::Failed
            );
        }
    }

    #[test]
    fn platform_slots_serialize_selector_transactions_across_vps() {
        let resources = PvResources::prepare(&platform_config(), 2, &[], 1 << 45)
            .unwrap()
            .unwrap();
        let backend = PlatformBackend::default();
        std::thread::scope(|scope| {
            for vp in 0..2 {
                let resources = &resources;
                let backend = &backend;
                scope.spawn(move || {
                    // SAFETY: prepare initialized both pages; each thread owns one protocol slot.
                    let page = unsafe {
                        &*resources.regions[0]
                            .as_ptr()
                            .cast::<u8>()
                            .add(vp as usize * pv::PAGE_SIZE)
                            .cast::<ControlPage>()
                    };
                    let mut client = Client::new(Identity {
                        generation: resources.descriptor.generation,
                        channel: 0,
                        vp,
                    });
                    client
                        .publish(
                            page,
                            Request {
                                service: Service::Control,
                                opcode: pv::OP_NEGOTIATE,
                                device: 0,
                                args: [0; 6],
                            },
                        )
                        .unwrap();
                    resources
                        .service_vp(vp, Some(backend), &Mutex::default(), None)
                        .unwrap();
                    assert_eq!(
                        client.complete(page).unwrap().values[1],
                        Service::Control.bit() | Service::Platform.bit()
                    );
                    for _ in 0..128 {
                        client
                            .publish(
                                page,
                                platform_request(pv::platform::IOAPIC_READ, vp as u64, 0),
                            )
                            .unwrap();
                        resources
                            .service_vp(vp, Some(backend), &Mutex::default(), None)
                            .unwrap();
                        let reply = client.complete(page).unwrap();
                        assert_eq!(reply.status, Status::Success);
                        assert_eq!(reply.values[0], u64::from(vp));
                    }
                });
            }
        });
        assert_eq!(backend.accesses.lock().unwrap().len(), 512);
    }

    #[test]
    fn device_memory_contains_only_the_actual_io_backing() {
        use vm_memory::{Bytes, GuestAddressSpace, GuestMemory, GuestMemoryRegion};

        let resources = PvResources::prepare(&platform_config(), 2, &[], 1 << 45)
            .unwrap()
            .unwrap();
        let transport = vm_memory::GuestMemoryAtomic::new(GuestMemoryMmap::new());
        let worker = transport.clone();
        assert_eq!(worker.memory().num_regions(), 0);
        transport
            .lock()
            .unwrap()
            .replace(resources.device_memory().unwrap());
        let memory = worker.memory();
        let io = &resources.regions[resources.descriptor.channels()[0].io as usize];
        let start = io.start_addr();
        let end = start.0 + io.len();
        assert_eq!(memory.num_regions(), 1);
        assert_eq!(memory.get_host_address(start).unwrap(), io.as_ptr());
        memory.write_slice(&[0x5a], start).unwrap();
        let mut value = [0];
        io.read_slice(&mut value, vm_memory::MemoryRegionAddress(0))
            .unwrap();
        assert_eq!(value, [0x5a]);
        assert!(memory.check_range(GuestAddress(end - 1), 1));
        assert!(!memory.check_range(GuestAddress(end - 1), 2));
        assert!(!memory.check_range(GuestAddress(u64::MAX), 2));
        for gpa in [
            0x1000,
            resources.regions[0].start_addr().0,
            resources.descriptor_gpa,
        ] {
            assert!(memory.read_slice(&mut value, GuestAddress(gpa)).is_err());
        }
        drop(resources);
        memory.read_slice(&mut value, start).unwrap();
        assert_eq!(value, [0x5a]);
    }

    #[test]
    fn confidential_aliases_preserve_backing_rights_and_private_descriptor() {
        let resources = PvResources::prepare(&platform_config(), 2, &[], 1 << 45)
            .unwrap()
            .unwrap();
        let plain = resources.memory_regions(0);
        let shared = resources.memory_regions(1 << 45);
        let regions = plain.len() - 1;
        assert_eq!(shared.len(), 2 * regions + 1);
        for i in 0..regions {
            assert_eq!(
                shared[i + regions].guest_pfn,
                plain[i].guest_pfn | (1 << 33)
            );
            assert_eq!(shared[i + regions].userspace_addr, plain[i].userspace_addr);
            assert_eq!(shared[i + regions].size, plain[i].size);
            assert_eq!(
                shared[i + regions].rights,
                THHV_MEM_R_READ | THHV_MEM_R_WRITE
            );
            assert_eq!(shared[i + regions].flags, THHV_MEM_F_ALIAS);
        }
        let descriptor = shared.last().unwrap();
        assert_eq!(descriptor.guest_pfn, resources.descriptor_gpa >> 12);
        assert_eq!(descriptor.rights, THHV_MEM_R_READ);
        assert_eq!(descriptor.flags, 0);
    }

    #[test]
    fn undeclared_platform_service_cannot_reach_devices() {
        let config = ThemisConfig::from_json_str(include_str!(
            "../../../../eunomia/policies/pv/standard.json"
        ))
        .unwrap();
        let resources = PvResources::prepare(&config, 1, &[], 1 << 45)
            .unwrap()
            .unwrap();
        let backend = PlatformBackend::default();
        // SAFETY: prepare initialized this page in the live owned mapping.
        let page = unsafe { &*resources.regions[0].as_ptr().cast::<ControlPage>() };
        let mut client = Client::new(Identity {
            generation: resources.descriptor.generation,
            channel: 0,
            vp: 0,
        });
        client
            .publish(page, platform_request(pv::platform::IOAPIC_READ, 1, 0))
            .unwrap();
        resources
            .service_vp(0, Some(&backend), &Mutex::default(), None)
            .unwrap();
        assert_eq!(client.complete(page).unwrap().status, Status::Unsupported);
        assert!(backend.accesses.lock().unwrap().is_empty());
    }

    #[test]
    fn mmap_backed_slots_dispatch_independently_on_the_owning_vp() {
        let config = ThemisConfig::from_json_str(include_str!(
            "../../../../eunomia/policies/pv/standard.json"
        ))
        .unwrap();
        let resources = PvResources::prepare(&config, 2, &[], 1 << 45)
            .unwrap()
            .unwrap();
        let mut clients = (0..2)
            .map(|vp| {
                Client::new(Identity {
                    generation: resources.descriptor.generation,
                    channel: 0,
                    vp,
                })
            })
            .collect::<Vec<_>>();
        let pages = (0..2)
            .map(|vp| {
                // SAFETY: prepare initialized both aligned pages in this live owned mapping.
                unsafe {
                    &*resources.regions[0]
                        .as_ptr()
                        .cast::<u8>()
                        .add(vp * pv::PAGE_SIZE)
                        .cast::<ControlPage>()
                }
            })
            .collect::<Vec<_>>();
        for sequence in 0..3 {
            for vp in 0..2 {
                clients[vp]
                    .publish(
                        pages[vp],
                        Request {
                            service: Service::Control,
                            opcode: pv::OP_ECHO,
                            device: 0,
                            args: [vp as u64, sequence, 0, 0, 0, 0],
                        },
                    )
                    .unwrap();
            }
            resources
                .service_vp(1, None, &Mutex::default(), None)
                .unwrap();
            assert_eq!(clients[0].complete(pages[0]), Err(pv::Error::NoCompletion));
            assert_eq!(clients[1].complete(pages[1]).unwrap().values[0], 1);
            resources
                .service_vp(0, None, &Mutex::default(), None)
                .unwrap();
            assert_eq!(clients[0].complete(pages[0]).unwrap().values[1], sequence);
            resources
                .service_vp(0, None, &Mutex::default(), None)
                .unwrap();
        }
        assert!(
            resources
                .service_vp(2, None, &Mutex::default(), None)
                .is_err()
        );
    }

    #[test]
    fn prepared_layout_requires_real_notification_authority_before_launch() {
        let config = ThemisConfig::from_json_str(include_str!(
            "../../../../eunomia/policies/pv/standard.json"
        ))
        .unwrap();
        let mut resources = PvResources::prepare(&config, 2, &[], 1 << 45)
            .unwrap()
            .unwrap();
        assert!(
            resources
                .descriptor
                .validate(resources.descriptor_gpa, 1 << 45)
                .is_err()
        );
        resources.descriptor.bind_doorbell(0, 7, 42).unwrap();
        resources
            .descriptor
            .validate(resources.descriptor_gpa, 1 << 45)
            .unwrap();
        assert!(resources.descriptor.bind_doorbell(0, 7, 43).is_err());
    }

    #[test]
    fn unsupported_service_and_device_fail_explicitly() {
        for (service, opcode, device) in [
            (Service::Platform, 0, 0),
            (Service::Control, pv::OP_ECHO, 1),
            (Service::Control, u32::MAX, 0),
        ] {
            assert_eq!(
                dispatch_control(
                    Request {
                        service,
                        opcode,
                        device,
                        args: [0; 6],
                    },
                    Service::Control.bit()
                )
                .status,
                Status::Unsupported
            );
        }
    }
}
