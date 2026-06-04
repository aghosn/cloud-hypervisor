//! Shared-anonymous `mmap` region wrapper used for THHV meta pages and
//! the per-vCPU comm page.  Owns the lifetime of the mapping and
//! `munmap`s on drop.

use std::ptr;

use libc::c_void;

pub(crate) struct MmapRegion {
    pub addr: *mut c_void,
    pub len: usize,
}

// SAFETY: The mapping lifetime is owned by the struct and access is coordinated by the VMM.
unsafe impl Send for MmapRegion {}
// SAFETY: The mapping points to shared memory supplied to the kernel; raw pointer sharing is intentional.
unsafe impl Sync for MmapRegion {}

impl MmapRegion {
    pub fn new_shared_anonymous(len: usize) -> anyhow::Result<Self> {
        // SAFETY: Valid mmap arguments for a shared anonymous region.
        let addr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }

        Ok(Self { addr, len })
    }

    pub fn as_u64(&self) -> u64 {
        self.addr as usize as u64
    }
}

impl Drop for MmapRegion {
    fn drop(&mut self) {
        if !self.addr.is_null() && self.addr != libc::MAP_FAILED {
            // SAFETY: Region was created by mmap with this address/length.
            unsafe {
                libc::munmap(self.addr, self.len);
            }
        }
    }
}
