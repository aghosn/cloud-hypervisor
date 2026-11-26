use std::io;
use std::os::fd::{AsRawFd, RawFd};

use vmm_sys_util::eventfd::EventFd;

use crate::IoEventAddress;
use crate::vm::DataMatch;

struct Registration {
    address: IoEventAddress,
    datamatch: Option<DataMatch>,
    source_fd: RawFd,
    event: EventFd,
}

#[derive(Default)]
pub(super) struct IoEvents(Vec<Registration>);

fn same_address(left: IoEventAddress, right: IoEventAddress) -> bool {
    match (left, right) {
        (IoEventAddress::Pio(a), IoEventAddress::Pio(b))
        | (IoEventAddress::Mmio(a), IoEventAddress::Mmio(b)) => a == b,
        _ => false,
    }
}

impl IoEvents {
    pub fn register(
        &mut self,
        fd: &EventFd,
        address: IoEventAddress,
        datamatch: Option<DataMatch>,
    ) -> io::Result<()> {
        if self.0.iter().any(|entry| {
            same_address(entry.address, address)
                && match (&entry.datamatch, &datamatch) {
                    (Some(DataMatch::DataMatch32(a)), Some(DataMatch::DataMatch32(b))) => a == b,
                    (Some(DataMatch::DataMatch64(a)), Some(DataMatch::DataMatch64(b))) => a == b,
                    (Some(_), Some(_)) => false,
                    _ => true,
                }
        }) {
            return Err(io::Error::from(io::ErrorKind::AlreadyExists));
        }
        self.0.push(Registration {
            address,
            datamatch,
            source_fd: fd.as_raw_fd(),
            event: fd.try_clone()?,
        });
        Ok(())
    }

    pub fn unregister(&mut self, fd: RawFd, address: IoEventAddress) -> io::Result<()> {
        let previous = self.0.len();
        self.0
            .retain(|entry| entry.source_fd != fd || !same_address(entry.address, address));
        if self.0.len() == previous {
            return Err(io::Error::from(io::ErrorKind::NotFound));
        }
        Ok(())
    }

    pub fn write(&self, address: IoEventAddress, data: &[u8]) -> io::Result<bool> {
        for entry in &self.0 {
            if !same_address(entry.address, address) {
                continue;
            }
            let matched = match entry.datamatch {
                None => true,
                Some(DataMatch::DataMatch32(value)) => data == value.to_le_bytes(),
                Some(DataMatch::DataMatch64(value)) => data == value.to_le_bytes(),
            };
            if matched {
                entry.event.write(1)?;
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_address_space_width_and_complete_value() {
        let event = EventFd::new(libc::EFD_NONBLOCK).unwrap();
        let mut events = IoEvents::default();
        let address = IoEventAddress::Mmio(0x1000);
        events
            .register(&event, address, Some(DataMatch::DataMatch64(1 << 40)))
            .unwrap();
        assert!(
            !events
                .write(IoEventAddress::Pio(0x1000), &(1u64 << 40).to_le_bytes())
                .unwrap()
        );
        assert!(!events.write(address, &0u32.to_le_bytes()).unwrap());
        assert!(!events.write(address, &1u64.to_le_bytes()).unwrap());
        assert!(events.write(address, &(1u64 << 40).to_le_bytes()).unwrap());
        assert_eq!(event.read().unwrap(), 1);
        assert!(
            events
                .write(IoEventAddress::Mmio(0x1001), &[0])
                .is_ok_and(|v| !v)
        );
    }

    #[test]
    fn retains_event_owner_and_removes_registration() {
        let event = EventFd::new(libc::EFD_NONBLOCK).unwrap();
        let observer = event.try_clone().unwrap();
        let fd = event.as_raw_fd();
        let address = IoEventAddress::Pio(0x80);
        let mut events = IoEvents::default();
        events.register(&event, address, None).unwrap();
        drop(event);
        assert!(events.write(address, &[7]).unwrap());
        assert_eq!(observer.read().unwrap(), 1);
        events.unregister(fd, address).unwrap();
        assert!(!events.write(address, &[7]).unwrap());
        assert_eq!(
            events.unregister(fd, address).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
    }

    #[test]
    fn rejects_overlaps_but_allows_distinct_matches() {
        let event = EventFd::new(libc::EFD_NONBLOCK).unwrap();
        let address = IoEventAddress::Mmio(0x1000);
        let mut events = IoEvents::default();
        events
            .register(&event, address, Some(DataMatch::DataMatch32(1)))
            .unwrap();
        events
            .register(&event, address, Some(DataMatch::DataMatch32(2)))
            .unwrap();
        assert_eq!(
            events.register(&event, address, None).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        assert!(events.write(address, &2u32.to_le_bytes()).unwrap());
        assert_eq!(event.read().unwrap(), 1);
        assert!(!events.write(address, &[2]).unwrap());
    }
}
