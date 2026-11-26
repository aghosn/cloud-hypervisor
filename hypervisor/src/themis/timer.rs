use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, ensure};
use vmm_sys_util::eventfd::EventFd;

use super::abi::{THHV_INJECT_INTERRUPT, ThhvInjectInterrupt};
use super::consts::CPUID_LEAF_TSC_FREQ;
use super::helpers::ioctl_with_mut_ref;

fn cpuid_tsc_khz(leaf: std::arch::x86_64::CpuidResult) -> Option<u32> {
    if leaf.eax == 0 || leaf.ebx == 0 || leaf.ecx == 0 {
        return None;
    }
    let khz = u64::from(leaf.ecx) * u64::from(leaf.ebx) / u64::from(leaf.eax) / 1000;
    u32::try_from(khz).ok().filter(|&value| value != 0)
}

fn read_tsc() -> u64 {
    // SAFETY: RDTSC/LFENCE are available on supported x86-64 hosts.
    unsafe {
        std::arch::x86_64::_mm_lfence();
        let value = std::arch::x86_64::_rdtsc();
        std::arch::x86_64::_mm_lfence();
        value
    }
}

pub(super) fn host_tsc_khz() -> anyhow::Result<u32> {
    static KHZ: OnceLock<Result<u32, String>> = OnceLock::new();
    KHZ.get_or_init(|| {
        if std::arch::x86_64::__cpuid(0).eax >= CPUID_LEAF_TSC_FREQ
            && let Some(khz) = cpuid_tsc_khz(std::arch::x86_64::__cpuid(CPUID_LEAF_TSC_FREQ))
        {
            return Ok(khz);
        }
        // A missing crystal frequency cannot be inferred from the CPU's
        // nominal core clock. Measure the actual TSC against a monotonic clock.
        let start = Instant::now();
        let ticks = read_tsc();
        std::thread::sleep(Duration::from_millis(50));
        let elapsed_ticks = read_tsc().checked_sub(ticks).ok_or("TSC moved backwards")?;
        let elapsed = start.elapsed().as_nanos();
        let khz = u128::from(elapsed_ticks) * 1_000_000 / elapsed;
        u32::try_from(khz)
            .ok()
            .filter(|&khz| khz != 0)
            .ok_or_else(|| "invalid measured TSC frequency".into())
    })
    .as_ref()
    .copied()
    .map_err(|error| anyhow!("{error}"))
}

fn timer_vector(lvt: u32) -> Option<u8> {
    (lvt & (1 << 16) == 0 && (lvt >> 17) & 3 == 2 && lvt & 0xff >= 16).then_some(lvt as u8)
}

pub(super) fn valid_lvt(lvt: u64) -> bool {
    lvt & !0x7_00ff == 0 && (lvt >> 17) & 3 != 3 && (lvt & (1 << 16) != 0 || lvt & 0xff >= 16)
}

pub(super) struct LapicTimer {
    fd: OwnedFd,
    stop: EventFd,
    lvt: Arc<Mutex<Option<u32>>>,
    worker: Option<JoinHandle<anyhow::Result<()>>>,
}

impl LapicTimer {
    pub fn new(partition: Arc<OwnedFd>, vp_index: u32) -> anyhow::Result<Self> {
        Self::with_inject(move |vector| {
            let mut request = ThhvInjectInterrupt {
                vp_index,
                vector,
                pad: [0; 3],
            };
            ioctl_with_mut_ref(partition.as_raw_fd(), THHV_INJECT_INTERRUPT, &mut request)
                .context("inject LAPIC timer interrupt")
        })
    }

    fn with_inject(
        inject: impl Fn(u8) -> anyhow::Result<()> + Send + 'static,
    ) -> anyhow::Result<Self> {
        // SAFETY: timerfd_create has no pointer arguments.
        let raw = unsafe {
            libc::timerfd_create(
                libc::CLOCK_MONOTONIC,
                libc::TFD_CLOEXEC | libc::TFD_NONBLOCK,
            )
        };
        if raw < 0 {
            return Err(io::Error::last_os_error().into());
        }
        // SAFETY: raw is a newly created, valid descriptor owned by this call.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        let timer = fd.try_clone()?;
        let stop = EventFd::new(libc::EFD_CLOEXEC | libc::EFD_NONBLOCK)?;
        let stop_reader = stop.try_clone()?;
        let lvt = Arc::new(Mutex::new(None));
        let configuration = lvt.clone();
        let worker = std::thread::Builder::new()
            .name("lapic-timer".into())
            .spawn(move || {
                let result = (|| -> anyhow::Result<()> {
                    let mut fds = [
                        libc::pollfd {
                            fd: stop_reader.as_raw_fd(),
                            events: libc::POLLIN,
                            revents: 0,
                        },
                        libc::pollfd {
                            fd: timer.as_raw_fd(),
                            events: libc::POLLIN,
                            revents: 0,
                        },
                    ];
                    loop {
                        // SAFETY: fds is a live array with exactly the supplied length.
                        if unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as _, -1) } < 0 {
                            let error = io::Error::last_os_error();
                            if error.kind() == io::ErrorKind::Interrupted {
                                continue;
                            }
                            return Err(error.into());
                        }
                        if fds[0].revents & libc::POLLIN != 0 {
                            return Ok(());
                        }
                        ensure!(
                            fds.iter().all(|fd| fd.revents & !libc::POLLIN == 0),
                            "timer poll failed"
                        );
                        if fds[1].revents & libc::POLLIN == 0 {
                            continue;
                        }
                        let lvt = configuration.lock().unwrap();
                        let mut expirations = 0u64;
                        // SAFETY: expirations is writable for all eight bytes.
                        let count = unsafe {
                            libc::read(timer.as_raw_fd(), (&mut expirations as *mut u64).cast(), 8)
                        };
                        if count < 0
                            && io::Error::last_os_error().kind() == io::ErrorKind::WouldBlock
                        {
                            continue;
                        }
                        ensure!(
                            count == 8,
                            "read LAPIC timer: {}",
                            io::Error::last_os_error()
                        );
                        if let Some(vector) = lvt.and_then(timer_vector) {
                            inject(vector)?;
                        }
                    }
                })();
                if let Err(error) = &result {
                    log::error!("Themis LAPIC timer stopped: {error:#}");
                }
                result
            })?;
        Ok(Self {
            fd,
            stop,
            lvt,
            worker: Some(worker),
        })
    }

    pub fn configure(&self, lvt: u32) -> anyhow::Result<()> {
        ensure!(valid_lvt(u64::from(lvt)), "invalid LAPIC timer LVT");
        *self.lvt.lock().unwrap() = Some(lvt);
        Ok(())
    }

    pub fn arm(&self, deadline: u64) -> anyhow::Result<()> {
        ensure!(
            !self
                .worker
                .as_ref()
                .is_some_and(|worker| worker.is_finished()),
            "LAPIC timer worker has stopped"
        );
        let configuration = self.lvt.lock().unwrap();
        if deadline != 0 {
            let lvt = configuration
                .ok_or_else(|| anyhow!("guest has not published its LAPIC timer LVT"))?;
            ensure!(
                (lvt >> 17) & 3 == 2,
                "LAPIC timer LVT {lvt:#x} is not in TSC-deadline mode"
            );
            ensure!(
                lvt & (1 << 16) != 0 || lvt & 0xff >= 16,
                "invalid LAPIC timer vector"
            );
        }
        let ns = if deadline == 0 {
            0
        } else {
            let khz = host_tsc_khz()?;
            let ticks = deadline.saturating_sub(read_tsc());
            u64::try_from(u128::from(ticks) * 1_000_000 / u128::from(khz))?.max(1)
        };
        let arm = libc::itimerspec {
            it_interval: libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            },
            it_value: libc::timespec {
                tv_sec: (ns / 1_000_000_000).try_into()?,
                tv_nsec: (ns % 1_000_000_000).try_into()?,
            },
        };
        // SAFETY: arm is initialized and the optional old-value pointer is null.
        if unsafe { libc::timerfd_settime(self.fd.as_raw_fd(), 0, &arm, std::ptr::null_mut()) } < 0
        {
            return Err(io::Error::last_os_error().into());
        }
        Ok(())
    }
}

impl Drop for LapicTimer {
    fn drop(&mut self) {
        if let Err(error) = self.stop.write(1) {
            log::error!("stop Themis LAPIC timer: {error}");
        }
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            log::error!("Themis LAPIC timer worker panicked");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_crystal_is_not_guessed() {
        let leaf = std::arch::x86_64::CpuidResult {
            eax: 1,
            ebx: 100,
            ecx: 0,
            edx: 0,
        };
        assert_eq!(cpuid_tsc_khz(leaf), None);
        assert_eq!(
            cpuid_tsc_khz(std::arch::x86_64::CpuidResult {
                ecx: 24_000_000,
                ..leaf
            }),
            Some(2_400_000)
        );
        assert_eq!(
            cpuid_tsc_khz(std::arch::x86_64::CpuidResult {
                ecx: 25_000_000,
                ..leaf
            }),
            Some(2_500_000)
        );
    }

    #[test]
    fn deadline_uses_guest_vector_and_honors_mask_and_disarm() {
        let (sender, receiver) = std::sync::mpsc::channel();
        let timer = LapicTimer::with_inject(move |vector| {
            sender.send(vector)?;
            Ok(())
        })
        .unwrap();
        timer.arm(1).unwrap_err();
        timer.configure(0x51).unwrap();
        timer.arm(1).unwrap_err();
        timer.configure((2 << 17) | 0x51).unwrap();
        timer.arm(1).unwrap();
        assert_eq!(receiver.recv_timeout(Duration::from_secs(2)).unwrap(), 0x51);
        timer.configure((2 << 17) | (1 << 16) | 0x61).unwrap();
        timer.arm(1).unwrap();
        receiver
            .recv_timeout(Duration::from_millis(30))
            .unwrap_err();
        timer.configure((2 << 17) | 0x71).unwrap();
        let deadline = read_tsc() + u64::from(host_tsc_khz().unwrap()) * 100;
        timer.arm(deadline).unwrap();
        timer.arm(0).unwrap();
        receiver
            .recv_timeout(Duration::from_millis(150))
            .unwrap_err();
        timer
            .arm(read_tsc() + u64::from(host_tsc_khz().unwrap()) * 50)
            .unwrap();
        timer.configure((2 << 17) | (1 << 16) | 0x71).unwrap();
        receiver
            .recv_timeout(Duration::from_millis(100))
            .unwrap_err();
        timer.configure((2 << 17) | 0x71).unwrap();
        timer.arm(1).unwrap();
        assert_eq!(receiver.recv_timeout(Duration::from_secs(2)).unwrap(), 0x71);
        drop(timer);
        receiver.recv().unwrap_err();
    }
}
