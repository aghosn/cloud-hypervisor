//! Free helper functions used across the Themis backend submodules:
//! THHV register-tuple builders, segment-register conversions, and thin
//! `ioctl(2)` wrappers.

use std::os::fd::{FromRawFd, OwnedFd};

use libc::c_ulong;

use super::abi::{THHV_QUERY, ThhvQuery, ThhvRegNameValue};
use super::consts::VpRegister;
use crate::arch::x86::SegmentRegister;

pub(super) fn reg(name: VpRegister, value: u64) -> ThhvRegNameValue {
    ThhvRegNameValue {
        name: name as u64,
        value,
    }
}

pub(super) fn segment_regs(
    sel: VpRegister,
    base: VpRegister,
    lim: VpRegister,
    ar: VpRegister,
    seg: &SegmentRegister,
) -> [ThhvRegNameValue; 4] {
    [
        reg(sel, seg.selector.into()),
        reg(base, seg.base),
        reg(lim, seg.limit.into()),
        reg(ar, segment_access_rights(seg).into()),
    ]
}

pub(super) fn segment_from_raw(selector: u64, base: u64, limit: u64, access_rights: u64) -> SegmentRegister {
    let ar = access_rights as u32;
    SegmentRegister {
        base,
        limit: limit as u32,
        selector: selector as u16,
        type_: (ar & 0x0f) as u8,
        s: ((ar >> 4) & 0x1) as u8,
        dpl: ((ar >> 5) & 0x3) as u8,
        present: ((ar >> 7) & 0x1) as u8,
        avl: ((ar >> 12) & 0x1) as u8,
        l: ((ar >> 13) & 0x1) as u8,
        db: ((ar >> 14) & 0x1) as u8,
        g: ((ar >> 15) & 0x1) as u8,
        unusable: ((ar >> 16) & 0x1) as u8,
    }
}

pub(super) fn segment_access_rights(seg: &SegmentRegister) -> u32 {
    u32::from(seg.type_ & 0x0f)
        | (u32::from(seg.s & 0x1) << 4)
        | (u32::from(seg.dpl & 0x3) << 5)
        | (u32::from(seg.present & 0x1) << 7)
        | (u32::from(seg.avl & 0x1) << 12)
        | (u32::from(seg.l & 0x1) << 13)
        | (u32::from(seg.db & 0x1) << 14)
        | (u32::from(seg.g & 0x1) << 15)
        | (u32::from(seg.unusable & 0x1) << 16)
}

pub(super) fn page_size() -> anyhow::Result<usize> {
    // SAFETY: sysconf is thread-safe for this name.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(page_size as usize)
}

pub(super) fn query_meta_pages(fd: i32, query_type: u32) -> anyhow::Result<u64> {
    let mut query = ThhvQuery {
        query_type,
        rsvd: 0,
        result: 0,
    };
    ioctl_with_mut_ref(fd, THHV_QUERY, &mut query)?;
    Ok(query.result)
}

pub(super) fn ioctl_noarg(fd: i32, request: c_ulong) -> std::io::Result<()> {
    // SAFETY: ioctl is called with a valid fd and no third argument.
    let ret = unsafe { libc::ioctl(fd, request) };
    if ret < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

pub(super) fn ioctl_with_mut_ref<T>(fd: i32, request: c_ulong, arg: &mut T) -> std::io::Result<()> {
    // SAFETY: ioctl is called with a valid fd and a pointer to a live repr(C) object.
    let ret = unsafe { libc::ioctl(fd, request, arg as *mut T) };
    if ret < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

pub(super) fn ioctl_with_mut_ref_ret_fd<T>(
    fd: i32,
    request: c_ulong,
    arg: &mut T,
) -> std::io::Result<OwnedFd> {
    // SAFETY: ioctl is called with a valid fd and a pointer to a live repr(C) object.
    let ret = unsafe { libc::ioctl(fd, request, arg as *mut T) };
    if ret < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        // SAFETY: ioctl returned a new owned fd.
        Ok(unsafe { OwnedFd::from_raw_fd(ret) })
    }
}
