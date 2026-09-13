#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bdxl_layerguard.py - keep backup data off the layer transitions of a
multi-layer Blu-ray (BDXL / BD-R DL / BD-R XL) disc.

A 100 GB BDXL is three 32,794,804,224-byte recording layers stacked on one
side.  The drive has to refocus at each layer change, and the couple of
hundred megabytes around those two crossover points are the least reliable
part of the disc.  This tool builds a staging tree whose *byte layout on the
finished disc* is known in advance, so that the only things sitting on the
transitions are zero-filled buffer files nobody cares about.

Three subcommands:

  build-udf   RECOMMENDED. Write a ready-to-burn UDF 1.02 image directly.
              This tool lays out every extent itself, so the buffers land on
              the transitions byte-exactly - no estimating, no second pass.
              The source tree is preserved unchanged (restore is a plain
              recursive copy), files of any size work (UDF has no 4 GiB
              limit), and a file that would land on a transition is split at
              the extent level so it steps over the buffer while still being
              one file to the reader. Needs no external tools.

  plan        The older ISO 9660 route: build a staging tree of segments,
              pad files and buffers, to be handed to xorriso. Kept because
              plain ISO 9660 + Rock Ridge is maximally boring to read, but
              it wastes space on alignment padding, splits the tree across
              segment directories, cannot represent a file over 4 GiB
              without multi-extent records that macOS and Windows mishandle,
              and its offsets depend on estimating the filesystem metadata.

  verify      Read the finished image (or the burned disc) and report which
              file each layer transition actually falls inside, with the
              margin on each side. Understands UDF and ISO 9660; --raw skips
              the filesystem entirely and just checks the bytes are zero,
              which works on any filesystem and on Windows raw devices.

Standard library only.  Python 3.8+.  Linux, macOS, Windows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SECTOR = 2048

# An ISO 9660 directory record carries a 32-bit data length, so one extent
# holds at most 4 GiB - 1, rounded down to a whole sector.  Bigger files become
# several records chained with the "not final extent" flag (ISO level 3).
ISO_EXTENT_MAX = 0xFFFFF800          # 4,294,965,248

# (layers, bytes per layer).  Layer sizes are the standard Blu-ray figures:
# 25,025,314,816 B per layer for BD-R/BD-RE, 32,794,804,224 B per layer for
# the XL formats.  100 GB = 3 x 32,794,804,224 = 98,384,412,672 B.
DISC_PRESETS: Dict[str, Tuple[int, int]] = {
    "bd-25":  (1, 25_025_314_816),
    "bd-50":  (2, 25_025_314_816),
    "bd-100": (3, 32_794_804_224),
    "bd-128": (4, 32_794_804_224),
}
DEFAULT_DISC = "bd-100"

BUFFER_PREFIX = "BUFF_T"
PAD_PREFIX = "PAD_T"
MANIFEST_NAME = "MANIFEST.TXT"


class LayoutError(Exception):
    """Raised when the requested backup + buffers cannot be laid out."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def align_up(n: int, unit: int = SECTOR) -> int:
    return ((n + unit - 1) // unit) * unit


def align_down(n: int, unit: int = SECTOR) -> int:
    return (n // unit) * unit


def parse_size(text: str) -> int:
    """'512M' / '1.5GiB' -> bytes.  K/M/G/T are binary; KB/MB/GB decimal."""
    s = str(text).strip().replace("_", "")
    if not s:
        raise ValueError("empty size")
    units = [
        ("KIB", 1024), ("MIB", 1024 ** 2), ("GIB", 1024 ** 3), ("TIB", 1024 ** 4),
        ("KB", 1000), ("MB", 1000 ** 2), ("GB", 1000 ** 3), ("TB", 1000 ** 4),
        ("K", 1024), ("M", 1024 ** 2), ("G", 1024 ** 3), ("T", 1024 ** 4),
        ("B", 1),
    ]
    up = s.upper()
    for suffix, mult in units:
        if up.endswith(suffix):
            num = up[: -len(suffix)].strip()
            if not num:
                raise ValueError("size %r has no number" % text)
            return int(round(float(num) * mult))
    return int(round(float(up)))


def human(n: int) -> str:
    if n < 0:
        return "-" + human(-n)
    for unit, div in (("TiB", 1024 ** 4), ("GiB", 1024 ** 3), ("MiB", 1024 ** 2), ("KiB", 1024)):
        if n >= div:
            return "%.2f %s" % (n / div, unit)
    return "%d B" % n


def both(n: int) -> str:
    return "%s (%s)" % (human(n), format(n, ","))


# ---------------------------------------------------------------------------
# scanning the source tree
# ---------------------------------------------------------------------------

@dataclass
class Entry:
    rel: str        # POSIX-style path relative to the source root
    size: int
    src: str        # absolute source path


@dataclass
class Scan:
    files: List[Entry] = field(default_factory=list)
    empty_dirs: List[str] = field(default_factory=list)
    n_dirs: int = 0                # every directory walked, root included
    total_bytes: int = 0
    padded_bytes: int = 0          # sum of sector-aligned sizes
    skipped_symlinks: List[str] = field(default_factory=list)
    skipped_special: List[str] = field(default_factory=list)
    unreadable: List[str] = field(default_factory=list)
    hardlink_groups: int = 0       # files sharing an inode with another file
    largest: int = 0
    # accumulated filesystem-metadata cost (see estimate_metadata)
    meta_iso: int = 0
    meta_joliet: int = 0
    ptable_iso: int = 0
    ptable_joliet: int = 0


def _rec_len(name_len: int) -> int:
    """ISO 9660 directory record length: 33 B header + name, padded to even."""
    n = 33 + name_len
    return n + (n & 1)


def scan_source(root: Path, follow_symlinks: bool = False) -> Scan:
    sc = Scan()
    root = root.resolve()
    inodes: Dict[Tuple[int, int], int] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirnames.sort()
        filenames.sort()
        d = Path(dirpath)
        sc.n_dirs += 1
        if not dirnames and not filenames and d != root:
            sc.empty_dirs.append(d.relative_to(root).as_posix())
        if not follow_symlinks:
            kept = []
            for name in dirnames:
                if os.path.islink(os.path.join(dirpath, name)):
                    sc.skipped_symlinks.append((d / name).relative_to(root).as_posix())
                else:
                    kept.append(name)
            dirnames[:] = kept

        # "." and ".." records open every directory extent
        iso_dir = 2 * _rec_len(1)
        jol_dir = 2 * _rec_len(1)

        for name in dirnames:
            iso_dir += _rec_len(min(len(name), 31)) + 90   # 90 ~ Rock Ridge
            jol_dir += _rec_len(2 * min(len(name), 64))
        # path table entries for this directory itself
        nm = len(d.name) if d != root else 1
        sc.ptable_iso += 8 + min(nm, 31) + (min(nm, 31) & 1)
        sc.ptable_joliet += 8 + 2 * min(nm, 64)

        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = (d / name).relative_to(root).as_posix()
            try:
                st = os.stat(full) if follow_symlinks else os.lstat(full)
            except OSError:
                sc.unreadable.append(rel)
                continue
            if stat.S_ISLNK(st.st_mode):
                sc.skipped_symlinks.append(rel)
                continue
            if not stat.S_ISREG(st.st_mode):
                sc.skipped_special.append(rel)
                continue
            if st.st_nlink > 1 and st.st_ino:
                key = (st.st_dev, st.st_ino)
                inodes[key] = inodes.get(key, 0) + 1
            sc.files.append(Entry(rel=rel, size=st.st_size, src=full))
            sc.total_bytes += st.st_size
            sc.padded_bytes += align_up(st.st_size)
            sc.largest = max(sc.largest, st.st_size)
            # files carry a ";1" version suffix in the ISO namespace, and a
            # file over 4 GiB is split into one record per 4 GiB extent
            extents = max(1, -(-st.st_size // ISO_EXTENT_MAX))
            iso_dir += extents * (_rec_len(min(len(name), 31) + 2) + 90)
            jol_dir += extents * _rec_len(2 * min(len(name), 64))

        sc.meta_iso += align_up(iso_dir)
        sc.meta_joliet += align_up(jol_dir)

    sc.hardlink_groups = sum(1 for v in inodes.values() if v > 1)
    return sc


# ---------------------------------------------------------------------------
# filesystem metadata estimate
# ---------------------------------------------------------------------------

def estimate_metadata(sc: Scan, joliet: bool = True, udf: bool = False,
                      safety: float = 0.10) -> int:
    """Estimate the bytes the writer puts in front of the first file extent.

    mkisofs-family writers emit, in this order: a 32 KiB system area, the
    volume descriptors, the path tables, the complete directory hierarchy
    (once per namespace), and only then file contents.  Directory extents are
    sector aligned, which is why this is computed per directory rather than
    per entry - a directory with six small entries still costs a whole sector
    in each namespace, and on a tree of git objects that term dominates
    everything else.

    The estimate is a first-pass guess.  `verify` reports the real figure from
    the built image; feed it back as --metadata-reserve for an exact layout.
    """
    entries = len(sc.files) + sc.n_dirs

    fixed = 16 * SECTOR          # system area
    fixed += 8 * SECTOR          # PVD, Joliet SVD, terminator, version block

    total = fixed + sc.meta_iso + 2 * align_up(sc.ptable_iso)
    if joliet:
        total += sc.meta_joliet + 2 * align_up(sc.ptable_joliet)

    if udf:
        # UDF allocates a file entry block per file/dir, plus its own anchors,
        # and xorriso does not necessarily place them all before the data.
        total += entries * SECTOR + 2 * 1024 * 1024

    return align_up(int(total * (1.0 + safety)))


# ---------------------------------------------------------------------------
# the layout
# ---------------------------------------------------------------------------

@dataclass
class Item:
    """One top-level entry of the staging tree, in on-disc order."""
    kind: str              # "segment" | "pad" | "buffer" | "manifest"
    name: str              # top-level name in the staging tree
    start: int             # planned byte offset on the disc
    size: int              # payload bytes
    padded: int            # bytes actually consumed (sector aligned)
    entries: List[Entry] = field(default_factory=list)
    transition: Optional[int] = None   # byte offset of the transition it guards


@dataclass
class Geometry:
    layers: int
    layer_bytes: int

    @property
    def capacity(self) -> int:
        return self.layers * self.layer_bytes

    @property
    def transitions(self) -> List[int]:
        return [self.layer_bytes * k for k in range(1, self.layers)]


def build_layout(sc: Scan, geo: Geometry, buffer_size: int, metadata_reserve: int,
                 bias: float = 0.5, manifest_reserve: int = 0) -> Tuple[List[Item], dict]:
    n_trans = len(geo.transitions)
    buffer_padded = align_up(buffer_size)

    required = metadata_reserve + sc.padded_bytes + n_trans * buffer_padded + manifest_reserve
    if required > geo.capacity:
        raise LayoutError(
            "does not fit on the disc.\n"
            "  backup data (sector aligned) : %s\n"
            "  %d x buffer                  : %s\n"
            "  filesystem metadata (est.)   : %s\n"
            "  manifest reserve             : %s\n"
            "  ------------------------------ \n"
            "  required                     : %s\n"
            "  disc capacity                : %s\n"
            "  short by                     : %s\n"
            "Reduce --buffer, drop data, or split the backup over two discs."
            % (both(sc.padded_bytes), n_trans, both(n_trans * buffer_padded),
               both(metadata_reserve), both(manifest_reserve),
               both(required), both(geo.capacity), both(required - geo.capacity))
        )

    placed = [False] * len(sc.files)
    items: List[Item] = []
    cursor = metadata_reserve
    seq = 1

    def fill(limit: int, cur: int) -> Tuple[List[Entry], int]:
        """Greedily take files, in source order, that still fit below limit."""
        chosen: List[Entry] = []
        for i, e in enumerate(sc.files):
            if placed[i]:
                continue
            need = align_up(e.size)
            if cur + need <= limit:
                placed[i] = True
                chosen.append(e)
                cur += need
        return chosen, cur

    unguarded: List[int] = []

    def add_segment(layer: int, start: int, end: int, entries: List[Entry]) -> None:
        nonlocal seq
        # skip empty middle segments, but always keep the first one: the
        # source's empty directories are recreated there
        if not entries and items:
            return
        items.append(Item(kind="segment", name="%02d_DATA_L%d" % (seq, layer),
                          start=start, size=end - start, padded=end - start,
                          entries=entries))
        seq += 1

    for idx, T in enumerate(geo.transitions, start=1):
        buf_start = align_down(T - int(buffer_size * bias))
        if not (buf_start < T < buf_start + buffer_padded):
            raise LayoutError(
                "with --bias %.2f the buffer would not actually cover the "
                "transition at %s. Use a bias between 0.05 and 0.95."
                % (bias, both(T)))
        if buf_start < cursor:
            raise LayoutError(
                "buffer #%d would have to start at %s, which is before the "
                "current write position %s - the data placed so far already "
                "runs past it. Use a smaller --buffer."
                % (idx, both(buf_start), both(cursor)))

        seg_entries, new_cursor = fill(buf_start, cursor)
        add_segment(idx, cursor, new_cursor, seg_entries)
        cursor = new_cursor

        # If the backup is already fully placed and ends well short of this
        # transition, the transition sits in blank space: no buffer needed, and
        # certainly no multi-gigabyte pad to reach it.  "Well short" means more
        # than one buffer width of slack, so that metadata drift cannot creep
        # the last file into the transition zone.
        if all(placed) and cursor + buffer_size < buf_start:
            unguarded.extend(geo.transitions[idx - 1:])
            break

        pad = buf_start - cursor
        if pad > 0:
            items.append(Item(kind="pad", name="%02d_%s%d.BIN" % (seq, PAD_PREFIX, idx),
                              start=cursor, size=pad, padded=pad, transition=T))
            seq += 1
            cursor += pad

        items.append(Item(kind="buffer", name="%02d_%s%d.BIN" % (seq, BUFFER_PREFIX, idx),
                          start=cursor, size=buffer_size, padded=buffer_padded,
                          transition=T))
        seq += 1
        cursor += buffer_padded
    else:
        # tail segment, everything after the last transition
        tail_entries, new_cursor = fill(geo.capacity - manifest_reserve, cursor)
        add_segment(geo.layers, cursor, new_cursor, tail_entries)
        cursor = new_cursor

    leftover = [e for i, e in enumerate(sc.files) if not placed[i]]
    if leftover:
        biggest = max(leftover, key=lambda e: e.size)
        raise LayoutError(
            "could not place %d file(s), %s in total.\n"
            "Largest unplaced: %s (%s).\n"
            "This happens when a single file is bigger than the space left "
            "between two transitions, or when the buffers eat the room the "
            "data needs. Try a smaller --buffer, or split that file."
            % (len(leftover), both(sum(align_up(e.size) for e in leftover)),
               biggest.rel, both(biggest.size))
        )

    if manifest_reserve:
        items.append(Item(kind="manifest", name="%02d_%s" % (seq, MANIFEST_NAME),
                          start=cursor, size=0, padded=manifest_reserve))

    stats = {
        "data_padded": sc.padded_bytes,
        "buffers": sum(i.padded for i in items if i.kind == "buffer"),
        "pads": sum(i.padded for i in items if i.kind == "pad"),
        "metadata_reserve": metadata_reserve,
        "manifest_reserve": manifest_reserve,
        "end_of_data": cursor + manifest_reserve,
        "free_tail": geo.capacity - (cursor + manifest_reserve),
        "unguarded_transitions": unguarded,
    }
    return items, stats


def verify_layout(items: List[Item], geo: Geometry) -> List[str]:
    """Re-derive the guarantee from the computed layout.  Belt and braces."""
    problems: List[str] = []
    cursor = None
    for it in items:
        if cursor is not None and it.start != cursor:
            problems.append("gap/overlap before %s: expected start %d, got %d"
                            % (it.name, cursor, it.start))
        cursor = it.start + it.padded
    if cursor is not None and cursor > geo.capacity:
        problems.append("layout ends at %d, past capacity %d" % (cursor, geo.capacity))

    end_of_layout = cursor or 0
    # every transition must be inside a buffer item, or past the end of the
    # written data (where there is nothing to lose)
    for T in geo.transitions:
        owner = None
        for it in items:
            if it.start <= T < it.start + it.padded:
                owner = it
                break
        if owner is None:
            if T < end_of_layout:
                problems.append("transition at %d is in an unaccounted gap" % T)
        elif owner.kind != "buffer":
            problems.append("transition at %d falls in %s (%s), not a buffer"
                            % (T, owner.name, owner.kind))

    # and no data file may straddle a transition
    for it in items:
        if it.kind != "segment":
            continue
        off = it.start
        for e in it.entries:
            end = off + align_up(e.size)
            for T in geo.transitions:
                if off < T < end:
                    problems.append("data file %s spans transition at %d" % (e.rel, T))
            off = end
    return problems


# ---------------------------------------------------------------------------
# materialising the staging tree
# ---------------------------------------------------------------------------

def make_zero_file(path: Path, size: int, sparse: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sparse:
        with open(path, "wb") as f:
            f.truncate(size)
        if path.stat().st_size == size:
            return
    chunk = b"\0" * (4 * 1024 * 1024)
    with open(path, "wb") as f:
        left = size
        while left > 0:
            n = min(left, len(chunk))
            f.write(chunk[:n])
            left -= n


def materialise(src: str, dst: Path, mode: str, counters: Dict[str, int]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    order = {
        "auto": ("hardlink", "symlink", "copy"),
        "hardlink": ("hardlink",),
        "symlink": ("symlink",),
        "copy": ("copy",),
    }[mode]
    last_err: Optional[Exception] = None
    for how in order:
        try:
            if how == "hardlink":
                os.link(src, dst)
            elif how == "symlink":
                os.symlink(os.path.abspath(src), dst)
            else:
                import shutil
                shutil.copy2(src, dst)
            counters[how] = counters.get(how, 0) + 1
            return
        except (OSError, NotImplementedError) as exc:
            last_err = exc
            continue
    raise LayoutError("cannot place %s into the staging tree: %s" % (src, last_err))


def write_manifest(path: Path, items: List[Item], sc: Scan, geo: Geometry,
                   cfg: argparse.Namespace, stats: dict) -> None:
    lines: List[str] = []
    w = lines.append
    w("bdxl_layerguard manifest")
    w("generated        : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    w("source           : %s" % cfg.source)
    w("disc             : %s  (%d layers x %s = %s)"
      % (cfg.disc, geo.layers, both(geo.layer_bytes), both(geo.capacity)))
    w("layer transitions: %s" % ", ".join(format(t, ",") for t in geo.transitions))
    w("buffer size      : %s" % both(cfg.buffer_bytes))
    w("files            : %d in %d directories, %s"
      % (len(sc.files), sc.n_dirs, both(sc.total_bytes)))
    w("")
    w("RESTORE: merge the DATA_L* directories into one tree; they keep the")
    w("original relative paths. The PAD_* and BUFF_* files are zero filled")
    w("and can be deleted. e.g. on Linux:")
    w("    for d in /mnt/disc/*_DATA_L*; do cp -a \"$d/.\" /restore/; done")
    w("")
    w("=== planned on-disc layout ===")
    w("%-22s %16s %16s %16s" % ("entry", "start", "end", "size"))
    for it in items:
        w("%-22s %16s %16s %16s"
          % (it.name, format(it.start, ","), format(it.start + it.padded, ","),
             format(it.padded, ",")))
    w("")
    w("=== planned file offsets (segment order; the writer may reorder")
    w("    files *within* a segment, which does not move segment bounds) ===")
    w("%16s %16s  %s" % ("start", "size", "path"))
    for it in items:
        if it.kind != "segment":
            continue
        off = it.start
        for e in it.entries:
            w("%16s %16s  %s/%s" % (format(off, ","), format(e.size, ","), it.name, e.rel))
            off += align_up(e.size)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plan_files(plan_dir: Path, items: List[Item], sc: Scan, geo: Geometry,
                     cfg: argparse.Namespace, stats: dict, staging: Path) -> None:
    plan_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(cfg.source),
        "staging": str(staging),
        "disc": cfg.disc,
        "layers": geo.layers,
        "layer_bytes": geo.layer_bytes,
        "capacity": geo.capacity,
        "transitions": geo.transitions,
        "buffer_bytes": cfg.buffer_bytes,
        "bias": cfg.bias,
        "metadata_reserve": cfg.metadata_reserve,
        "stats": stats,
        "items": [
            {"kind": it.kind, "name": it.name, "start": it.start, "size": it.size,
             "padded": it.padded, "transition": it.transition,
             "file_count": len(it.entries)}
            for it in items
        ],
    }
    (plan_dir / "layout.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Sort-weight files: higher weight = closer to the start of the image.
    # Only the top level needs pinning, because the total padded size of a
    # segment does not depend on the order of the files inside it.
    # xorriso wants "NUMBER PATH", genisoimage wants "PATH NUMBER".
    step = 1000
    xo: List[str] = []
    gi: List[str] = []
    for i, it in enumerate(items):
        w = step * (len(items) + 1 - i)
        xo.append("%d /%s" % (w, it.name))
        gi.append("%s %d" % ((staging / it.name).as_posix(), w))
    (plan_dir / "sort_weights_xorriso.txt").write_text("\n".join(xo) + "\n", encoding="utf-8")
    (plan_dir / "sort_weights_genisoimage.txt").write_text("\n".join(gi) + "\n", encoding="utf-8")

    label = cfg.label or ("BACKUP_" + time.strftime("%Y%m%d"))
    iso_out = cfg.iso or "backup.iso"
    # NB: xorriso/libisofs has no UDF writer at all - "-as mkisofs -udf" is
    # rejected as an unsupported option. Only the metadata estimate knows
    # about UDF, for when the image is built by a tool that can write it.
    ns = "-iso-level 3 -r" + ("" if cfg.no_joliet else " -J -joliet-long")
    cmds = f"""# ---------------------------------------------------------------------------
# 1) build the image  (xorriso, recommended)
# ---------------------------------------------------------------------------
# -iso-level 3 gives multi-extent support for files > 4 GiB, so no -udf is
# needed; UDF metadata placement is less predictable, so leave it off unless
# you specifically need it (and then use a bigger buffer).
xorriso -as mkisofs \\
  {ns} \\
  -V "{label}" \\
  --sort-weight-list "{(plan_dir / 'sort_weights_xorriso.txt').as_posix()}" \\
  -o "{iso_out}" \\
  "{staging.as_posix()}"

# same thing with genisoimage instead (note -no-cache-inodes: without it,
# hardlinked duplicates share one extent and every offset after them shifts)
# genisoimage {ns} -no-cache-inodes \\
#   -sort "{(plan_dir / 'sort_weights_genisoimage.txt').as_posix()}" \\
#   -V "{label}" -o "{iso_out}" "{staging.as_posix()}"

# ---------------------------------------------------------------------------
# 2) PROVE the layout before burning  <-- do not skip this
# ---------------------------------------------------------------------------
python3 "{Path(__file__).name}" verify "{iso_out}" --disc {cfg.disc} \\
  --buffer {cfg.buffer_bytes}

# ---------------------------------------------------------------------------
# 3) burn as a single closed session (SAO/DAO). No multisession, no packet
#    writing - either would move LBA 0 and shift every offset.
# ---------------------------------------------------------------------------
xorriso -as cdrecord -v dev=/dev/sr0 -sao -eject "{iso_out}"
#   or: growisofs -dvd-compat -Z /dev/sr0="{iso_out}"
#   Windows: burn "{iso_out}" with ImgBurn in "Write image file to disc" mode,
#            Write Type = SAO/DAO, Verify on.
#   macOS:   hdiutil burn "{iso_out}"

# ---------------------------------------------------------------------------
# 4) after burning, verify against the disc itself
#    Linux/macOS only - on Windows, verify the .iso in step 2 instead, since
#    Python cannot do raw sector reads of \\\\.\\D: reliably.
# ---------------------------------------------------------------------------
python3 "{Path(__file__).name}" verify /dev/sr0 --disc {cfg.disc} --buffer {cfg.buffer_bytes}
#   macOS: the raw device is /dev/rdisk<N> (see `diskutil list`)

# ---------------------------------------------------------------------------
# 5) restoring later
# ---------------------------------------------------------------------------
# The backup tree is split across the *_DATA_L* directories, each keeping the
# original relative paths. Merge them and the original tree is back:
#   for d in /mnt/disc/*_DATA_L*; do cp -a "$d/." /restore/; done
# The *_PAD_*.BIN and *_BUFF_*.BIN files are zeros and can be ignored.
"""
    (plan_dir / "burn_commands.txt").write_text(cmds, encoding="utf-8")


# ---------------------------------------------------------------------------
# ISO 9660 reader, for verify
# ---------------------------------------------------------------------------

@dataclass
class IsoFile:
    path: str
    lba: int
    size: int          # sum of all extents
    padded: int
    extents: List[Tuple[int, int]] = field(default_factory=list)


# A filler file is one this tool created: a numbered PAD_Tn.BIN / BUFF_Tn.BIN
# sitting at the root of the image.  Anchoring both the pattern and the depth
# means a file inside the backup data cannot be mistaken for a buffer.
_FILLER_RE = re.compile(r"^\d+_(?:%s|%s)\d+\.BIN\.?$"
                        % (re.escape(BUFFER_PREFIX), re.escape(PAD_PREFIX)))


def is_filler(iso_path: str) -> bool:
    if iso_path.count("/") != 1:          # must be at the volume root
        return False
    base = iso_path.rsplit("/", 1)[-1]
    return bool(_FILLER_RE.match(base.upper()) or _BUFFER_NAME_RE.match(base))


def _find_pvd(f) -> bytes:
    for sector in range(16, 64):
        f.seek(sector * SECTOR)
        buf = f.read(SECTOR)
        if len(buf) < 7:
            break
        if buf[1:6] == b"CD001":
            if buf[0] == 1:
                return buf
            if buf[0] == 255:
                break
    raise LayoutError("no ISO 9660 primary volume descriptor found "
                      "(is this a single-session ISO 9660 image?)")


def _records(data: bytes) -> Iterable[bytes]:
    i = 0
    while i < len(data):
        ln = data[i]
        if ln == 0:
            i = (i // SECTOR + 1) * SECTOR
            continue
        if i + ln > len(data):
            break
        yield data[i:i + ln]
        i += ln


def read_iso(path: str) -> Tuple[List[IsoFile], int, int]:
    """Return (files, volume_size_bytes, first_data_offset)."""
    files: List[IsoFile] = []
    with open(path, "rb") as f:
        pvd = _find_pvd(f)
        block = int.from_bytes(pvd[128:130], "little") or SECTOR
        vol_blocks = int.from_bytes(pvd[80:84], "little")
        root = pvd[156:190]
        root_lba = int.from_bytes(root[2:6], "little")
        root_len = int.from_bytes(root[10:14], "little")

        stack: List[Tuple[str, int, int]] = [("", root_lba, root_len)]
        seen = set()
        while stack:
            prefix, lba, length = stack.pop()
            if (lba, length) in seen:
                continue
            seen.add((lba, length))
            f.seek(lba * block)
            data = f.read(length)
            pending: Optional[IsoFile] = None
            for rec in _records(data):
                name_len = rec[32]
                raw = rec[33:33 + name_len]
                flags = rec[25]
                ext_lba = int.from_bytes(rec[2:6], "little")
                ext_len = int.from_bytes(rec[10:14], "little")
                if name_len == 1 and raw in (b"\x00", b"\x01"):
                    continue
                name = raw.decode("latin-1")
                if name.endswith(";1"):
                    name = name[:-2]
                full = prefix + "/" + name
                if flags & 0x02:
                    stack.append((full, ext_lba, ext_len))
                    continue
                if pending is not None and pending.path == full:
                    pending.size += ext_len
                    pending.padded += align_up(ext_len, block)
                else:
                    if pending is not None:
                        files.append(pending)
                    pending = IsoFile(path=full, lba=ext_lba, size=ext_len,
                                      padded=align_up(ext_len, block))
                if not (flags & 0x80):
                    files.append(pending)
                    pending = None
            if pending is not None:
                files.append(pending)

    files = [x for x in files if x.size > 0]
    files.sort(key=lambda x: x.lba)
    first_data = files[0].lba * SECTOR if files else 0
    return files, vol_blocks * block, first_data


def zero_run_around(path: str, T: int, window: int, block: int = 1 << 20) -> Tuple[int, int]:
    """Find how far the all-zero run containing offset T extends, by reading
    the image/disc directly.  Filesystem agnostic: works on UDF, on hybrid
    images, and on a burned disc, including Windows raw devices, because every
    read is sector aligned.  Returns (first_zero_byte, first_nonzero_after)."""
    window = align_up(window)
    lo = hi = align_down(T)
    with open(path, "rb", buffering=0) as f:
        pos = align_down(T)
        floor = max(0, align_down(T) - window)
        while pos > floor:
            start = max(floor, pos - block)
            f.seek(start)
            buf = f.read(pos - start)
            if not buf:
                break
            trimmed = buf.rstrip(b"\0")
            if trimmed:
                lo = start + len(trimmed)
                break
            lo = start
            pos = start
        pos = align_down(T)
        ceiling = pos + window
        while pos < ceiling:
            f.seek(pos)
            buf = f.read(min(block, ceiling - pos))
            if not buf:
                break
            trimmed = buf.lstrip(b"\0")
            if trimmed:
                hi = pos + (len(buf) - len(trimmed))
                break
            hi = pos + len(buf)
            pos += len(buf)
    return lo, hi, lo <= floor, hi >= ceiling


def cmd_raw_verify(cfg: argparse.Namespace) -> int:
    """Prove the transitions are zero by reading the bytes, not the metadata."""
    geo = geometry_from(cfg)
    window = parse_size(cfg.window) if cfg.window else \
        (parse_size(cfg.buffer) if cfg.buffer else 256 * 1024 * 1024)
    print("raw zero-run check on %s" % cfg.image)
    print("scanning up to %s either side of each transition\n" % human(window))
    ok = True
    for T in geo.transitions:
        lo, hi, hit_lo, hit_hi = zero_run_around(cfg.image, T, window)
        print("layer transition at %s" % both(T))
        if hi <= align_down(T):
            print("    verdict    : *** FAIL *** non-zero bytes at the transition")
            ok = False
        else:
            print("    zero run   : %s .. %s" % (format(lo, ","), format(hi, ",")))
            print("    margin     : %s%s of zeros before, %s%s after"
                  % (human(T - lo), " (scan limit)" if hit_lo else "",
                     human(hi - T), " (scan limit)" if hit_hi else ""))
            print("    verdict    : PASS - nothing but zeros across the transition")
        print()
    print("RESULT: %s" % ("PASS - all transitions sit in zero-filled space"
                          if ok else "FAIL - see above"))
    print("\nNote: this check cannot tell a zero-filled buffer from genuinely "
          "zero-filled\nbackup data. It is the right check for UDF images and "
          "burned discs; for an\nISO 9660 image, the default check names the "
          "actual file at each transition.")
    return 0 if ok else 2


def detect_filesystem(path: str) -> str:
    """'udf', 'iso9660', 'hybrid' or 'unknown', from the on-disc structures."""
    with open(path, "rb", buffering=0) as f:
        f.seek(16 * SECTOR)
        vrs = f.read(SECTOR)
        f.seek(256 * SECTOR)
        anchor = f.read(SECTOR)
    has_iso = len(vrs) >= 6 and vrs[1:6] == b"CD001"
    has_udf = (len(anchor) >= 2
               and struct.unpack_from("<H", anchor, 0)[0] == 2
               and (sum(anchor[0:4]) + sum(anchor[5:16])) & 0xFF == anchor[4])
    if has_iso and has_udf:
        return "hybrid"
    return "udf" if has_udf else ("iso9660" if has_iso else "unknown")


def cmd_verify(cfg: argparse.Namespace) -> int:
    if cfg.raw:
        return cmd_raw_verify(cfg)
    geo = geometry_from(cfg)
    kind = cfg.fs if cfg.fs != "auto" else detect_filesystem(cfg.image)
    if kind == "hybrid":
        print("NOTE: this image carries BOTH UDF and ISO 9660. If any file "
              "exceeds 4 GiB\n      the ISO 9660 side may report a truncated "
              "size, so which filesystem a\n      reader picks changes what it "
              "restores. Checking the UDF side.\n")
        kind = "udf"
    try:
        if kind == "udf":
            files, vol_size, first_data = read_udf(cfg.image)
        elif kind == "iso9660":
            files, vol_size, first_data = read_iso(cfg.image)
        else:
            raise LayoutError("no UDF or ISO 9660 structures found in %s" % cfg.image)
    except LayoutError as exc:
        print("%s\nFalling back to the raw zero-run check.\n" % exc)
        return cmd_raw_verify(cfg)
    print("filesystem       : %s" % kind)

    print("image            : %s" % cfg.image)
    print("volume size      : %s" % both(vol_size))
    print("disc             : %s  %d layers x %s = %s"
          % (cfg.disc, geo.layers, both(geo.layer_bytes), both(geo.capacity)))
    intervals = []
    for x in files:
        if x.extents:
            for off, ln in x.extents:
                intervals.append((off, off + align_up(ln), x))
        else:
            intervals.append((x.lba * SECTOR, x.lba * SECTOR + x.padded, x))
    data_end = max((e for _, e, _ in intervals), default=0)
    print("data files found : %d (%d extents)" % (len(files), len(intervals)))
    print("last data byte   : %s" % both(data_end))
    print("metadata before  : %s   <-- pass this to `plan --metadata-reserve`"
          % both(first_data))
    print("first file extent: %s" % (files[0].path if files else "(none)"))
    print()

    if vol_size > geo.capacity:
        print("FAIL  image is %s larger than the disc" % both(vol_size - geo.capacity))

    ok = True
    for T in geo.transitions:
        owner = None
        owner_range = (0, 0)
        for start, end, x in intervals:
            if start <= T < end:
                owner, owner_range = x, (start, end)
                break
        print("layer transition at %s" % both(T))
        if owner is None:
            if T >= data_end:
                print("    verdict    : PASS - %s past the end of the recorded "
                      "data, nothing there to lose" % human(T - data_end))
            else:
                print("    verdict    : *** FAIL *** offset falls in a gap no "
                      "file accounts for; the layout is not what was planned")
                ok = False
            print()
            continue
        start, end = owner_range
        print("    covered by : %s" % owner.path)
        print("    file range : %s .. %s (%s)"
              % (format(start, ","), format(end, ","), human(owner.padded)))
        print("    margin     : %s before the transition, %s after"
              % (human(T - start), human(end - T)))
        if is_filler(owner.path):
            print("    verdict    : PASS - zero-filled buffer, no backup data here")
            if cfg.buffer:
                want = parse_size(cfg.buffer)
                thin = min(T - start, end - T)
                if thin < want * 0.25:
                    print("    note       : only %s of buffer on one side. Re-run "
                          "`plan` with --metadata-reserve %d to centre it."
                          % (human(thin), first_data))
        else:
            print("    verdict    : *** FAIL *** real backup data sits on the transition")
            ok = False
        print()

    # anything straddling a transition at all
    straddlers = [(s, x) for (s, e, x) in intervals
                  for T in geo.transitions
                  if s < T < e and not is_filler(x.path)]
    if straddlers:
        print("data extents straddling a transition:")
        for s, x in straddlers:
            print("    %s  (%s at %s)" % (x.path, human(x.size), format(s, ",")))
        ok = False

    print("RESULT: %s" % ("PASS - every layer transition is covered by a buffer file"
                          if ok else "FAIL - see above"))
    return 0 if ok else 2


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def geometry_from(cfg: argparse.Namespace) -> Geometry:
    layers, layer_bytes = DISC_PRESETS[cfg.disc]
    if getattr(cfg, "layers", None):
        layers = cfg.layers
    if getattr(cfg, "layer_bytes", None):
        layer_bytes = parse_size(cfg.layer_bytes)
    layer_bytes = align_down(layer_bytes)
    if layers < 1 or layer_bytes < SECTOR:
        raise LayoutError("nonsensical disc geometry")
    return Geometry(layers=layers, layer_bytes=layer_bytes)


def cmd_plan(cfg: argparse.Namespace) -> int:
    source = Path(cfg.source).expanduser()
    if not source.is_dir():
        raise LayoutError("source folder does not exist: %s" % source)

    geo = geometry_from(cfg)
    cfg.buffer_bytes = parse_size(cfg.buffer)
    if cfg.buffer_bytes <= 0:
        raise LayoutError("--buffer must be positive")
    if geo.layers < 2:
        raise LayoutError("%s has one recording layer, so there are no layer "
                          "transitions to protect."
                          % ("the requested geometry" if cfg.layers else cfg.disc))
    if cfg.buffer_bytes >= geo.layer_bytes:
        raise LayoutError("--buffer (%s) must be smaller than one layer (%s)"
                          % (both(cfg.buffer_bytes), both(geo.layer_bytes)))
    if not 0.05 <= cfg.bias <= 0.95:
        raise LayoutError("--bias must be between 0.05 and 0.95 (0.5 centres "
                          "the buffer on the transition)")

    print("scanning %s ..." % source)
    sc = scan_source(source, follow_symlinks=cfg.follow_symlinks)
    if not sc.files:
        raise LayoutError("no regular files found under %s" % source)

    if cfg.metadata_reserve and str(cfg.metadata_reserve).startswith("@"):
        probe = str(cfg.metadata_reserve)[1:]
        _, _, first_data = read_iso(probe)
        meta = align_up(first_data)
        meta_note = "measured from %s" % probe
    elif cfg.metadata_reserve:
        meta = align_up(parse_size(cfg.metadata_reserve))
        meta_note = "given"
    else:
        meta = estimate_metadata(sc, joliet=not cfg.no_joliet, udf=cfg.udf)
        meta_note = "estimated"

    # the manifest is written last, in the tail, so an over-estimate here only
    # costs unused tail space and cannot move any transition
    manifest_reserve = 0 if cfg.no_manifest else align_up(8192 + 220 * (len(sc.files) + 64))

    items, stats = build_layout(sc, geo, cfg.buffer_bytes, meta,
                                bias=cfg.bias, manifest_reserve=manifest_reserve)
    problems = verify_layout(items, geo)
    if problems:
        raise LayoutError("internal layout check failed:\n  " + "\n  ".join(problems))

    # ---- report -----------------------------------------------------------
    print()
    print("source        : %s" % source)
    print("               %d files in %d dirs, %s (%s sector aligned)"
          % (len(sc.files), sc.n_dirs, both(sc.total_bytes), human(sc.padded_bytes)))
    if sc.largest:
        print("               largest file %s" % both(sc.largest))
    print("disc          : %s - %d layers x %s, capacity %s"
          % (cfg.disc, geo.layers, both(geo.layer_bytes), both(geo.capacity)))
    print("transitions   : %s" % ", ".join(both(t) for t in geo.transitions))
    print("buffer        : %s each, %.0f%% of it before the transition"
          % (both(cfg.buffer_bytes), cfg.bias * 100))
    print("metadata      : %s (%s)" % (both(meta), meta_note))
    print()
    print("%-22s %17s %17s %14s  %s" % ("entry", "start", "end", "size", "contents"))
    for it in items:
        if it.kind == "segment":
            desc = "%d files" % len(it.entries)
        elif it.kind == "buffer":
            desc = "zeros - guards transition at %s" % format(it.transition, ",")
        elif it.kind == "pad":
            desc = "zeros - alignment"
        else:
            desc = "layout manifest"
        print("%-22s %17s %17s %14s  %s"
              % (it.name, format(it.start, ","), format(it.start + it.padded, ","),
                 human(it.padded), desc))
    print()
    print("space         : data %s | buffers %s | alignment pads %s | "
          "metadata %s | unused tail %s"
          % (human(stats["data_padded"]), human(stats["buffers"]),
             human(stats["pads"]), human(meta), human(stats["free_tail"])))

    end_of_data = stats["end_of_data"]
    for T in geo.transitions:
        owner = next((i for i in items if i.start <= T < i.start + i.padded), None)
        if owner is None:
            print("check         : transition %s is %s past the end of the written "
                  "data - nothing there to lose, no buffer needed"
                  % (format(T, ","), human(T - end_of_data)))
        else:
            print("check         : transition %s is inside %s  (%s before / %s after)"
                  % (format(T, ","), owner.name, human(T - owner.start),
                     human(owner.start + owner.padded - T)))

    warn = []
    if sc.skipped_symlinks:
        warn.append("%d symlink(s) skipped (use --follow-symlinks to include "
                    "their targets)" % len(sc.skipped_symlinks))
    if sc.skipped_special:
        warn.append("%d non-regular file(s) skipped" % len(sc.skipped_special))
    if sc.unreadable:
        warn.append("%d path(s) could not be stat'ed" % len(sc.unreadable))
    if sc.hardlink_groups:
        warn.append("%d hardlinked file group(s) in the source: pass "
                    "-no-cache-inodes to genisoimage (xorriso is fine) or the "
                    "offsets will shift" % sc.hardlink_groups)
    if cfg.buffer_bytes < 64 * 1024 * 1024:
        warn.append("buffer is smaller than 64 MiB; metadata estimation drift "
                    "could eat it. 256 MiB - 1 GiB is a sane range for a 100 GB disc")
    if stats["pads"] > geo.capacity // 100:
        warn.append("%s went into alignment padding. That happens when a large "
                    "file cannot fit in the space left before a transition; "
                    "splitting the biggest files would recover most of it"
                    % human(stats["pads"]))
    if stats["unguarded_transitions"]:
        warn.append("%d transition(s) need no buffer: the backup ends before "
                    "them" % len(stats["unguarded_transitions"]))
    for m in warn:
        print("warning       : %s" % m)

    huge = [e for e in sc.files if e.size > ISO_EXTENT_MAX]
    if huge:
        print()
        print("*** %d file(s) are larger than the ISO 9660 extent limit of %s."
              % (len(huge), human(ISO_EXTENT_MAX)))
        print("    Largest: %s (%s)"
              % (max(huge, key=lambda e: e.size).rel,
                 human(max(e.size for e in huge))))
        print("    They will be stored as multi-extent files: one directory")
        print("    record per %s chunk, all pointing at one contiguous run of"
              % human(ISO_EXTENT_MAX))
        print("    bytes. The layout in this plan is unaffected - the chunks are")
        print("    contiguous and the total is the same - but READING them back")
        print("    is another matter:")
        print("      Linux + xorriso  : fine, one whole file")
        print("      macOS            : shows one entry per chunk, same name;")
        print("                         you cannot copy the file back correctly")
        print("      Windows CDFS     : may refuse the file outright")
        print("    Fix it one of two ways before you burn:")
        print("      1. split the big files below %s and plan again" % human(ISO_EXTENT_MAX))
        print("         (split -b 3900M big.mkv big.mkv.part_  /  cat parts > big.mkv)")
        print("      2. build a UDF image instead - UDF has no 4 GiB limit.")
        print("         xorriso cannot write UDF at all, so use one of:")
        print("           macOS  : hdiutil makehybrid -udf -udf-volume-name V \\")
        print("                      -o out.iso <staging>")
        print("           Windows: ImgBurn Build mode, File System = UDF 2.50+,")
        print("                      with 'sort files by source list order'")
        print("           Linux  : genisoimage -udf ...")
        print("         then check it with:  verify <iso> --raw --buffer %s"
              % cfg.buffer)
        print("         (--raw reads the bytes instead of the ISO 9660 metadata,")
        print("          which is the only way to check a UDF image here)")

    if cfg.dry_run:
        print("\n(dry run - nothing written)")
        return 0

    # ---- materialise ------------------------------------------------------
    staging = Path(cfg.out).expanduser().resolve()
    if staging.exists() and any(staging.iterdir()):
        raise LayoutError("staging folder %s exists and is not empty" % staging)
    plan_dir = Path(str(staging) + "_plan")
    staging.mkdir(parents=True, exist_ok=True)

    print("\nbuilding staging tree in %s" % staging)
    counters: Dict[str, int] = {}
    done = 0
    for it in items:
        if it.kind == "segment":
            base = staging / it.name
            base.mkdir(parents=True, exist_ok=True)
            for e in it.entries:
                materialise(e.src, base / Path(e.rel), cfg.link_mode, counters)
                done += 1
                if done % 2000 == 0:
                    print("  %d/%d files ..." % (done, len(sc.files)))
        elif it.kind in ("pad", "buffer"):
            make_zero_file(staging / it.name, it.size, sparse=not cfg.no_sparse)
        # manifest written below

    # empty dirs go into the first segment so the tree round-trips
    first_seg = next(i for i in items if i.kind == "segment")
    for d in sc.empty_dirs:
        (staging / first_seg.name / Path(d)).mkdir(parents=True, exist_ok=True)

    manifest_item = next((i for i in items if i.kind == "manifest"), None)
    if manifest_item is not None:
        write_manifest(staging / manifest_item.name, items, sc, geo, cfg, stats)

    write_plan_files(plan_dir, items, sc, geo, cfg, stats, staging)
    if manifest_item is not None:
        import shutil
        shutil.copy2(staging / manifest_item.name, plan_dir / MANIFEST_NAME)

    print("  placed %d files (%s)"
          % (done, ", ".join("%s=%d" % kv for kv in sorted(counters.items()))))
    print("\nwrote:")
    print("  %s   <- feed this to xorriso" % staging)
    print("  %s/layout.json" % plan_dir)
    print("  %s/sort_weights_xorriso.txt (+ genisoimage variant)" % plan_dir)
    print("  %s/burn_commands.txt   <- copy/paste from here" % plan_dir)
    print("\nNext: build the ISO with the command in burn_commands.txt, then run")
    print("  %s verify <iso> --disc %s --buffer %s"
          % (Path(sys.argv[0]).name, cfg.disc, cfg.buffer))
    print("before you burn. Verify reads the real directory records, so it is")
    print("the step that actually proves the buffers landed on the transitions.")
    return 0


# ---------------------------------------------------------------------------
# UDF 1.02 writer
#
# Why write the filesystem ourselves instead of driving mkisofs: because then
# the byte layout is not an estimate.  Every extent is placed by us, so the
# buffers land exactly on the transitions - no metadata guessing, no sort
# weights, no drift, no second pass.  Two further things fall out of it for
# free:
#
#   * The source tree is preserved as-is.  No 01_DATA_L1/ segment split, so a
#     restore is a plain recursive copy.
#   * A file that would land on a transition is split at the extent level -
#     UDF's allocation descriptors let one file live in several pieces - so it
#     steps over the buffer and continues after it.  No alignment padding is
#     wasted, and files of any size work: UDF stores lengths in 64 bits, so
#     the ISO 9660 4 GiB limit is gone.
#
# UDF revision 1.02 with a single physical (type 1) partition and plain File
# Entries is deliberately the most boring profile that exists - it is what
# DVD-Video uses, so every reader on Windows, macOS and Linux handles it.
# ---------------------------------------------------------------------------

UDF_MAX_EXTENT = 0x3FFFF800        # 1,073,739,776 - 30-bit length, block aligned
UDF_PART_START = 288               # first partition block, after the anchor at 256
UDF_IMPL_ID = "*bdxl_layerguard"
BUFFER_FILE_FMT = "_LAYER_BUFFER_%d.BIN"
UDF_MANIFEST = "_LAYER_MANIFEST.TXT"
_BUFFER_NAME_RE = re.compile(r"^_LAYER_BUFFER_\d+\.BIN$", re.I)


def _make_crc_table() -> List[int]:
    table = []
    for i in range(256):
        r = i << 8
        for _ in range(8):
            r = ((r << 1) ^ 0x1021) & 0xFFFF if r & 0x8000 else (r << 1) & 0xFFFF
        table.append(r)
    return table


_CRC_TABLE = _make_crc_table()


def crc_itu_t(data: bytes) -> int:
    """CRC-ITU-T (x^16+x^12+x^5+1, initial 0), as ECMA-167 requires."""
    crc = 0
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[((crc >> 8) ^ b) & 0xFF]
    return crc


def tagged(ident: int, location: int, body: bytes, serial: int = 1,
           version: int = 2) -> bytes:
    """Wrap a descriptor body in its 16-byte descriptor tag."""
    tag = bytearray(16)
    struct.pack_into("<HH", tag, 0, ident, version)
    struct.pack_into("<H", tag, 6, serial)
    struct.pack_into("<HH", tag, 8, crc_itu_t(body), len(body))
    struct.pack_into("<I", tag, 12, location)
    tag[4] = (sum(tag[0:4]) + sum(tag[5:16])) & 0xFF
    return bytes(tag) + body


def cs0(text: str) -> bytes:
    """OSTA CS0 compressed unicode: a marker byte then 8- or 16-bit chars."""
    try:
        return b"\x08" + text.encode("latin-1")
    except UnicodeEncodeError:
        return b"\x10" + text.encode("utf-16-be")


def dstring(text: str, size: int) -> bytes:
    enc = cs0(text) if text else b""
    while len(enc) > size - 1:            # trim whole characters, not bytes
        text = text[:-1]
        enc = cs0(text) if text else b""
    out = bytearray(size)
    out[0:len(enc)] = enc
    out[size - 1] = len(enc)
    return bytes(out)


def charspec() -> bytes:
    return b"\x00" + b"OSTA Compressed Unicode".ljust(63, b"\x00")


def regid(ident: str, suffix: bytes = b"", flags: int = 0) -> bytes:
    return (bytes([flags]) + ident.encode("ascii").ljust(23, b"\0")[:23]
            + suffix.ljust(8, b"\0")[:8])


DOMAIN_SUFFIX = struct.pack("<HBB", 0x0102, 0, 0) + b"\0" * 4   # UDF 1.02
UDF_SUFFIX = struct.pack("<HBB", 0x0102, 0, 0) + b"\0" * 4
IMPL_SUFFIX = bytes([0, 0]) + b"\0" * 6


def udf_time(when: Optional[float] = None) -> bytes:
    tm = time.gmtime(when if when is not None else time.time())
    return struct.pack("<HhBBBBBBBB", 0x1000, tm.tm_year, tm.tm_mon, tm.tm_mday,
                       tm.tm_hour, tm.tm_min, tm.tm_sec, 0, 0, 0)


def extent_ad(length: int, location: int) -> bytes:
    return struct.pack("<II", length, location)


def long_ad(length: int, block: int, part: int = 0, unique: int = 0) -> bytes:
    return (struct.pack("<IIH", length, block, part)
            + struct.pack("<HI", 0, unique & 0xFFFFFFFF))


def short_ad(length: int, block: int) -> bytes:
    return struct.pack("<II", length, block)


def align4(n: int) -> int:
    return (n + 3) & ~3


@dataclass(eq=False, repr=False)   # identity comparison: nodes point at their
class UNode:                       # parents, so generated __eq__/__repr__ recurse
    name: str
    is_dir: bool
    size: int = 0
    src: Optional[str] = None          # source file on disk
    data: Optional[bytes] = None       # in-memory content (manifest, buffers)
    zeros: bool = False                # write as zeros, no source
    children: List["UNode"] = field(default_factory=list)
    parent: Optional["UNode"] = None
    fe_block: int = 0                  # partition block of its File Entry
    dir_block: int = 0                 # partition block of its FID stream
    dir_blocks: int = 0
    fid_len: int = 0                   # bytes of FID stream
    unique: int = 0
    extents: List[Tuple[int, int]] = field(default_factory=list)  # (abs_off, len)

    def path(self) -> str:
        parts = []
        n: Optional[UNode] = self
        while n is not None and n.parent is not None:
            parts.append(n.name)
            n = n.parent
        return "/" + "/".join(reversed(parts))


def build_tree(sc: Scan) -> UNode:
    root = UNode(name="", is_dir=True)
    index: Dict[str, UNode] = {"": root}

    def ensure_dir(rel: str) -> UNode:
        if rel in index:
            return index[rel]
        head, _, tail = rel.rpartition("/")
        parent = ensure_dir(head) if rel != tail else root
        node = UNode(name=tail, is_dir=True, parent=parent)
        parent.children.append(node)
        index[rel] = node
        return node

    for d in sc.empty_dirs:
        ensure_dir(d)
    for e in sc.files:
        head, _, name = e.rel.rpartition("/")
        parent = ensure_dir(head) if head else root
        parent.children.append(UNode(name=name, is_dir=False, size=e.size,
                                     src=e.src, parent=parent))
    return root


def _fid_size(l_fi: int, l_iu: int) -> int:
    return align4(38 + l_fi + l_iu)


def plan_fid_streams(root: UNode) -> Tuple[List[UNode], List[UNode]]:
    """Size every directory's FID stream.  A File Identifier Descriptor is
    padded via its ImplementationUse field so that none of them straddles a
    logical block boundary - valid either way, but some readers insist."""
    dirs: List[UNode] = []
    files: List[UNode] = []
    stack = [root]
    while stack:
        n = stack.pop()
        (dirs if n.is_dir else files).append(n)
        if n.is_dir:
            n.children.sort(key=lambda c: (not c.is_dir, c.name))
            stack.extend(reversed(n.children))

    for d in dirs:
        lens = [_fid_size(0, 0)]                      # the parent entry
        for c in d.children:
            lens.append(_fid_size(len(cs0(c.name)), 0))
        pos = 0
        pads = [0] * len(lens)
        for i, ln in enumerate(lens):
            if (pos % SECTOR) + ln > SECTOR:
                slack = SECTOR - (pos % SECTOR)
                pads[i - 1] += slack
                pos += slack
            pos += ln
        d.fid_len = pos
        d.dir_blocks = max(1, -(-pos // SECTOR))
        d._fid_pads = pads                            # type: ignore[attr-defined]
    return dirs, files


def assign_metadata_blocks(dirs: List[UNode], files: List[UNode]) -> int:
    """Lay out FSD, terminator, directory entries/streams and file entries.
    Returns the number of partition blocks the metadata occupies."""
    blk = 2                                           # block 0 FSD, block 1 TD
    uid = 16
    for d in dirs:
        d.fe_block = blk
        blk += 1
        d.dir_block = blk
        blk += d.dir_blocks
        d.unique = 0 if d.parent is None else uid
        if d.parent is not None:
            uid += 1
    for fl in files:
        fl.fe_block = blk
        blk += 1
        fl.unique = uid
        uid += 1
    return blk


def render_fids(d: UNode) -> bytes:
    out = bytearray()
    pads: List[int] = getattr(d, "_fid_pads", [0] * (len(d.children) + 1))
    parent = d.parent or d
    entries = [(0x08, "", parent.fe_block, parent.unique)]
    for c in d.children:
        entries.append((0x02 if c.is_dir else 0x00, c.name, c.fe_block, c.unique))
    for i, (chars, name, blk, uniq) in enumerate(entries):
        fi = cs0(name) if name else b""
        l_iu = pads[i] if i < len(pads) else 0
        body = struct.pack("<HBB", 1, chars, len(fi))
        body += long_ad(SECTOR, blk, 0, uniq)
        body += struct.pack("<H", l_iu) + b"\0" * l_iu + fi
        body += b"\0" * (align4(38 + len(fi) + l_iu) - (38 + len(fi) + l_iu))
        out += tagged(257, d.dir_block + (len(out) // SECTOR), body)
    return bytes(out)


def render_file_entry(n: UNode, part_start: int) -> bytes:
    icb = struct.pack("<IHHHBB", 0, 4, 0, 1, 0, 4 if n.is_dir else 5)
    icb += struct.pack("<IH", 0, 0) + struct.pack("<H", 0)
    if n.is_dir:
        ads = short_ad(n.fid_len, n.dir_block)
        info_len = n.fid_len
        blocks = n.dir_blocks
        perms = 0x14A5
        links = 1 + sum(1 for c in n.children if c.is_dir)
    else:
        ads = b"".join(short_ad(ln, (off // SECTOR) - part_start)
                       for off, ln in n.extents)
        info_len = n.size
        blocks = sum(align_up(ln) // SECTOR for _, ln in n.extents)
        perms = 0x1084
        links = 1
    if len(ads) > SECTOR - 176:
        raise LayoutError("%s needs %d extents, too many for one file entry"
                          % (n.path(), len(ads) // 8))
    ts = udf_time()
    body = icb
    body += struct.pack("<III", 0xFFFFFFFF, 0xFFFFFFFF, perms)
    body += struct.pack("<HBBI", links, 0, 0, 0)
    body += struct.pack("<QQ", info_len, blocks)
    body += ts + ts + ts
    body += struct.pack("<I", 1)
    body += long_ad(0, 0)
    body += regid(UDF_IMPL_ID, IMPL_SUFFIX)
    body += struct.pack("<Q", n.unique)
    body += struct.pack("<II", 0, len(ads))
    body += ads
    return tagged(261, n.fe_block, body)


def udf_descriptors(vol_id: str, part_start: int, part_len: int,
                    fsd_block: int, root_block: int, n_files: int, n_dirs: int,
                    total_blocks: int) -> Dict[int, bytes]:
    """Every fixed descriptor, keyed by absolute sector."""
    out: Dict[int, bytes] = {}
    for i, ident in enumerate(("BEA01", "NSR02", "TEA01")):
        out[16 + i] = (bytes([0]) + ident.encode("ascii") + bytes([1])).ljust(SECTOR, b"\0")

    lv_id = vol_id
    vset = "%08x%s" % (int(time.time()) & 0xFFFFFFFF, vol_id)

    pvd = struct.pack("<II", 1, 0)
    pvd += dstring(vol_id, 32)
    pvd += struct.pack("<HHHH", 1, 1, 2, 2)
    pvd += struct.pack("<II", 1, 1)
    pvd += dstring(vset, 128)
    pvd += charspec() + charspec()
    pvd += extent_ad(0, 0) + extent_ad(0, 0)
    pvd += regid(UDF_IMPL_ID, IMPL_SUFFIX)
    pvd += udf_time()
    pvd += regid(UDF_IMPL_ID, IMPL_SUFFIX)
    pvd += b"\0" * 64
    pvd += struct.pack("<IH", 0, 0) + b"\0" * 22

    iuvd = struct.pack("<I", 2) + regid("*UDF LV Info", UDF_SUFFIX)
    iu = charspec() + dstring(lv_id, 128)
    iu += dstring("", 36) * 3
    iu += regid(UDF_IMPL_ID, IMPL_SUFFIX) + b"\0" * 128
    iuvd += iu

    pd = struct.pack("<IHH", 3, 1, 0)
    pd += regid("+NSR02")
    pd += b"\0" * 128
    pd += struct.pack("<I", 1)                     # access type: read-only
    pd += struct.pack("<II", part_start, part_len)
    pd += regid(UDF_IMPL_ID, IMPL_SUFFIX)
    pd += b"\0" * 128 + b"\0" * 156

    lvd = struct.pack("<I", 4) + charspec() + dstring(lv_id, 128)
    lvd += struct.pack("<I", SECTOR)
    lvd += regid("*OSTA UDF Compliant", DOMAIN_SUFFIX)
    lvd += long_ad(SECTOR, fsd_block, 0)
    lvd += struct.pack("<II", 6, 1)
    lvd += regid(UDF_IMPL_ID, IMPL_SUFFIX) + b"\0" * 128
    lvd += extent_ad(2 * SECTOR, 64)
    lvd += bytes([1, 6]) + struct.pack("<HH", 1, 0)

    usd = struct.pack("<II", 5, 0)
    td = b"\0" * 496

    for base in (32, 48):
        loc = base
        for ident, body in ((1, pvd), (4, iuvd), (5, pd), (6, lvd), (7, usd), (8, td)):
            out[loc] = tagged(ident, loc, body).ljust(SECTOR, b"\0")
            loc += 1

    lvid = udf_time() + struct.pack("<I", 1) + extent_ad(0, 0)
    lvid += struct.pack("<Q", 16 + n_files + n_dirs) + b"\0" * 24
    lvid += struct.pack("<II", 1, 46)
    lvid += struct.pack("<I", 0)                   # free space
    lvid += struct.pack("<I", part_len)
    lvid += regid(UDF_IMPL_ID, IMPL_SUFFIX)
    lvid += struct.pack("<IIHHH", n_files, n_dirs, 0x0102, 0x0102, 0x0102)
    out[64] = tagged(9, 64, lvid).ljust(SECTOR, b"\0")
    out[65] = tagged(8, 65, b"\0" * 496).ljust(SECTOR, b"\0")

    avdp = extent_ad(16 * SECTOR, 32) + extent_ad(16 * SECTOR, 48) + b"\0" * 480
    for loc in (256, total_blocks - 257, total_blocks - 1):
        if loc > 256 or loc == 256:
            out[loc] = tagged(2, loc, avdp).ljust(SECTOR, b"\0")

    fsd = udf_time() + struct.pack("<HH", 3, 3) + struct.pack("<II", 1, 1)
    fsd += struct.pack("<II", 0, 0)
    fsd += charspec() + dstring(lv_id, 128)
    fsd += charspec() + dstring(vol_id, 32)
    fsd += dstring("", 32) + dstring("", 32)
    fsd += long_ad(SECTOR, root_block, 0)
    fsd += regid("*OSTA UDF Compliant", DOMAIN_SUFFIX)
    fsd += long_ad(0, 0) + long_ad(0, 0) + b"\0" * 32
    out[part_start + fsd_block] = tagged(256, fsd_block, fsd).ljust(SECTOR, b"\0")
    out[part_start + fsd_block + 1] = tagged(8, fsd_block + 1, b"\0" * 496).ljust(SECTOR, b"\0")
    return out


def place_udf_data(files: List[UNode], buffers: List[UNode], data_start: int,
                   geo: Geometry, buffer_size: int, bias: float,
                   manifest: Optional[UNode]) -> Tuple[int, List[int]]:
    """Assign absolute byte extents.  Buffers are pinned to the transitions;
    file data flows around them, split at extent level where it has to be."""
    windows = []
    for T in geo.transitions:
        start = align_down(T - int(buffer_size * bias))
        windows.append((start, T))
    cursor = data_start
    used_buffers: List[int] = []
    wi = 0
    limit_total = geo.capacity - 258 * SECTOR       # keep room for end anchors

    def emit_buffer(idx: int) -> None:
        nonlocal cursor
        buffers[idx].extents = [(cursor, buffer_size)]
        buffers[idx].size = buffer_size
        cursor += align_up(buffer_size)
        used_buffers.append(idx)

    queue = list(files) + ([manifest] if manifest is not None else [])
    for n in queue:
        remaining = n.size
        if remaining == 0:
            n.extents = []
            continue
        while remaining > 0:
            if wi < len(windows) and cursor >= windows[wi][0]:
                emit_buffer(wi)
                wi += 1
                continue
            limit = windows[wi][0] if wi < len(windows) else limit_total
            avail = limit - cursor
            if avail <= 0 or (remaining > avail and align_down(avail) == 0):
                if wi < len(windows):
                    emit_buffer(wi)
                    wi += 1
                    continue
                raise LayoutError("ran out of room at %s" % both(cursor))
            chunk = min(remaining, avail, UDF_MAX_EXTENT)
            if chunk < remaining:
                chunk = align_down(chunk)           # non-final extents must be
                if chunk == 0:                      # whole blocks
                    if wi < len(windows):
                        emit_buffer(wi)
                        wi += 1
                        continue
                    raise LayoutError("ran out of room at %s" % both(cursor))
            n.extents.append((cursor, chunk))
            cursor += align_up(chunk)
            remaining -= chunk
            if cursor > limit_total:
                raise LayoutError(
                    "the data does not fit: needed past %s, disc holds %s"
                    % (both(cursor), both(geo.capacity)))
    # any transition still ahead of the data needs no buffer, unless the data
    # stops within one buffer width of it
    while wi < len(windows):
        if cursor + buffer_size >= windows[wi][0]:
            emit_buffer(wi)
        wi += 1
    return cursor, used_buffers


def cmd_build_udf(cfg: argparse.Namespace) -> int:
    source = Path(cfg.source).expanduser()
    if not source.is_dir():
        raise LayoutError("source folder does not exist: %s" % source)
    geo = geometry_from(cfg)
    if geo.layers < 2:
        raise LayoutError("%s has one recording layer, so there are no layer "
                          "transitions to protect."
                          % ("the requested geometry" if cfg.layers else cfg.disc))
    buffer_size = align_up(parse_size(cfg.buffer))
    if buffer_size <= 0 or buffer_size >= geo.layer_bytes:
        raise LayoutError("--buffer must be positive and smaller than one layer (%s)"
                          % both(geo.layer_bytes))
    if not 0.05 <= cfg.bias <= 0.95:
        raise LayoutError("--bias must be between 0.05 and 0.95")

    print("scanning %s ..." % source)
    sc = scan_source(source, follow_symlinks=cfg.follow_symlinks)
    if not sc.files:
        raise LayoutError("no regular files found under %s" % source)

    n_trans = len(geo.transitions)
    required = sc.padded_bytes + n_trans * buffer_size
    if required > geo.capacity:
        raise LayoutError(
            "does not fit on the disc.\n"
            "  backup data (block aligned) : %s\n"
            "  %d x buffer                 : %s\n"
            "  ----------------------------- \n"
            "  required                    : %s\n"
            "  disc capacity               : %s\n"
            "  short by                    : %s\n"
            "Reduce --buffer, drop data, or split across two discs."
            % (both(sc.padded_bytes), n_trans, both(n_trans * buffer_size),
               both(required), both(geo.capacity), both(required - geo.capacity)))

    # ---- build the tree, adding the buffer files and the manifest ----------
    root = build_tree(sc)
    buffers = [UNode(name=BUFFER_FILE_FMT % (i + 1), is_dir=False, zeros=True,
                     parent=root) for i in range(n_trans)]
    root.children.extend(buffers)
    manifest = None
    if not cfg.no_manifest:
        manifest = UNode(name=UDF_MANIFEST, is_dir=False, parent=root)
        root.children.append(manifest)

    dirs, files = plan_fid_streams(root)
    meta_blocks = assign_metadata_blocks(dirs, files)
    data_start = (UDF_PART_START + meta_blocks) * SECTOR
    real_files = [f for f in files if f not in buffers and f is not manifest]

    # pass one: place with an empty manifest to learn every offset
    end, used = place_udf_data(real_files, buffers, data_start, geo,
                               buffer_size, cfg.bias, None)
    if manifest is not None:
        text = render_udf_manifest(source, geo, buffer_size, real_files,
                                   buffers, used, end, cfg)
        manifest.data = text.encode("utf-8")
        manifest.size = len(manifest.data)
        for f in real_files:
            f.extents = []
        for b in buffers:
            b.extents = []
            b.size = 0
        end, used = place_udf_data(real_files, buffers, data_start, geo,
                                   buffer_size, cfg.bias, manifest)

    total_blocks = (align_up(end) // SECTOR) + 258
    part_len = total_blocks - 258 - UDF_PART_START + 1

    # ---- report ------------------------------------------------------------
    print()
    print("source        : %s" % source)
    print("               %d files in %d dirs, %s"
          % (len(sc.files), sc.n_dirs, both(sc.total_bytes)))
    print("disc          : %s - %d layers x %s, capacity %s"
          % (cfg.disc, geo.layers, both(geo.layer_bytes), both(geo.capacity)))
    print("filesystem    : UDF 1.02, single physical partition")
    print("metadata      : %s exactly (%d blocks: %d dirs, %d file entries)"
          % (both(meta_blocks * SECTOR), meta_blocks, len(dirs), len(files)))
    print("data starts   : %s" % both(data_start))
    print("image size    : %s" % both(total_blocks * SECTOR))
    print()
    for i, b in enumerate(buffers):
        T = geo.transitions[i]
        if i not in used:
            print("transition %d  : %s - past the end of the data, no buffer needed"
                  % (i + 1, format(T, ",")))
            continue
        s, ln = b.extents[0]
        print("transition %d  : %s" % (i + 1, format(T, ",")))
        print("               %s covers %s .. %s"
              % (b.name, format(s, ","), format(s + ln, ",")))
        print("               %s before / %s after  <- exact, not estimated"
              % (human(T - s), human(s + ln - T)))
    split = [f for f in real_files if len(f.extents) > 1]
    if split:
        print()
        print("%d file(s) split across a buffer or the 1 GiB extent limit; each "
              "is still\n              one file to the reader:" % len(split))
        for f in split[:5]:
            print("               %s -> %d extents" % (f.path(), len(f.extents)))
        if len(split) > 5:
            print("               ... and %d more" % (len(split) - 5))
    if sc.skipped_symlinks:
        print("warning       : %d symlink(s) skipped" % len(sc.skipped_symlinks))
    print()
    print("space         : data %s | buffers %s | metadata %s | unused tail %s"
          % (human(sc.padded_bytes), human(len(used) * buffer_size),
             human(meta_blocks * SECTOR),
             human(geo.capacity - total_blocks * SECTOR)))

    if cfg.dry_run:
        print("\n(dry run - no image written)")
        return 0

    out = Path(cfg.out).expanduser()
    write_udf_image(out, root, dirs, files, data_start, total_blocks, part_len,
                    meta_blocks, cfg, sparse=not cfg.no_sparse)
    print("\nwrote %s (%s)" % (out, both(out.stat().st_size)))
    print("\nNext:")
    print("  1. check it:  %s verify \"%s\" --disc %s --buffer %s"
          % (Path(sys.argv[0]).name, out, cfg.disc, cfg.buffer))
    print("  2. burn it with ImgBurn in 'Write image file to disc' mode")
    print("     (NOT Build mode), SAO/DAO, Verify ticked, slowest speed.")
    print("  3. restore is a plain recursive copy - the tree is unchanged, and")
    print("     the %s files are zeros you can ignore." % (BUFFER_FILE_FMT % 0)[:-5])
    return 0


def render_udf_manifest(source: Path, geo: Geometry, buffer_size: int,
                        files: List[UNode], buffers: List[UNode],
                        used: List[int], end: int,
                        cfg: argparse.Namespace) -> str:
    lines = ["bdxl_layerguard - UDF image manifest",
             "generated        : %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
             "source           : %s" % source,
             "disc             : %s (%d layers x %s)"
             % (cfg.disc, geo.layers, format(geo.layer_bytes, ",")),
             "layer transitions: %s" % ", ".join(format(t, ",") for t in geo.transitions),
             "buffer size      : %s each" % format(buffer_size, ","),
             "files            : %d" % len(files),
             "",
             "The directory tree on this disc is identical to the source, so a",
             "restore is just a recursive copy. The _LAYER_BUFFER_*.BIN files are",
             "zero filled and exist only to occupy the layer transition zones,",
             "which are the least reliable part of a multi-layer disc; delete them",
             "after restoring. Files listed with several extents are stored in",
             "pieces that step around a buffer - the filesystem presents them as",
             "one file, nothing extra is needed to read them.",
             "",
             "%18s %16s  %s" % ("start", "size", "path")]
    for b, i in ((buffers[i], i) for i in used):
        s, ln = b.extents[0]
        lines.append("%18s %16s  /%s" % (format(s, ","), format(ln, ","), b.name))
    for f in files:
        if not f.extents:
            lines.append("%18s %16s  %s" % ("-", 0, f.path()))
            continue
        lines.append("%18s %16s  %s"
                     % (format(f.extents[0][0], ","), format(f.size, ","), f.path()))
        for s, ln in f.extents[1:]:
            lines.append("%18s %16s    (continued)" % (format(s, ","), format(ln, ",")))
    return "\n".join(lines) + "\n"


def write_udf_image(out: Path, root: UNode, dirs: List[UNode], files: List[UNode],
                    data_start: int, total_blocks: int, part_len: int,
                    meta_blocks: int, cfg: argparse.Namespace,
                    sparse: bool = True) -> None:
    vol_id = (cfg.label or "BACKUP_" + time.strftime("%Y%m%d"))[:30]
    print("\nwriting %s ..." % out)
    with open(out, "wb") as f:
        fixed = udf_descriptors(vol_id, UDF_PART_START, part_len, 0,
                                root.fe_block, len(files), len(dirs),
                                total_blocks)
        for sector, blob in fixed.items():
            f.seek(sector * SECTOR)
            f.write(blob)
        for d in dirs:
            f.seek((UDF_PART_START + d.fe_block) * SECTOR)
            f.write(render_file_entry(d, UDF_PART_START).ljust(SECTOR, b"\0"))
            f.seek((UDF_PART_START + d.dir_block) * SECTOR)
            f.write(render_fids(d).ljust(d.dir_blocks * SECTOR, b"\0"))
        for fl in files:
            f.seek((UDF_PART_START + fl.fe_block) * SECTOR)
            f.write(render_file_entry(fl, UDF_PART_START).ljust(SECTOR, b"\0"))

        zero = b"\0" * (1 << 20)
        done = 0
        next_report = 1 << 30
        total_data = sum(n.size for n in files)
        for n in files:
            if not n.extents:
                continue
            if n.data is not None:
                for off, ln in n.extents:
                    f.seek(off)
                    f.write(n.data[:ln])
                continue
            if n.zeros:
                if sparse:
                    off, ln = n.extents[-1]
                    f.seek(off + ln - 1)
                    f.write(b"\0")
                else:
                    for off, ln in n.extents:
                        f.seek(off)
                        left = ln
                        while left:
                            k = min(left, len(zero))
                            f.write(zero[:k])
                            left -= k
                done += n.size
                continue
            with open(n.src, "rb") as src:
                for off, ln in n.extents:
                    f.seek(off)
                    left = ln
                    while left:
                        chunk = src.read(min(left, 1 << 22))
                        if not chunk:
                            raise LayoutError(
                                "%s got shorter while being read - do not change "
                                "the source folder during a build" % n.src)
                        f.write(chunk)
                        left -= len(chunk)
                    done += ln
            if done >= next_report:
                print("  copied %s / %s" % (human(done), human(total_data)))
                next_report += 1 << 30
        f.truncate(total_blocks * SECTOR)
        f.seek((total_blocks - 1) * SECTOR)
        f.write(fixed[total_blocks - 1])


# ---------------------------------------------------------------------------
# UDF reader, for verify
# ---------------------------------------------------------------------------

def read_udf(path: str) -> Tuple[List[IsoFile], int, int]:
    """Walk a UDF volume and return every file's real byte extents."""
    with open(path, "rb", buffering=0) as f:
        def sector(n: int) -> bytes:
            f.seek(n * SECTOR)
            return f.read(SECTOR)

        size = os.path.getsize(path)
        last = size // SECTOR - 1
        avdp = None
        for cand in (256, last, last - 256):
            if cand < 0:
                continue
            blob = sector(cand)
            if len(blob) >= 16 and struct.unpack_from("<H", blob, 0)[0] == 2:
                avdp = blob
                break
        if avdp is None:
            raise LayoutError("no UDF anchor volume descriptor pointer found")
        main_len, main_loc = struct.unpack_from("<II", avdp, 16)

        part_start = part_len = None
        fsd_block = None
        for i in range(main_len // SECTOR):
            blob = sector(main_loc + i)
            if len(blob) < 16:
                break
            ident = struct.unpack_from("<H", blob, 0)[0]
            if ident == 5:                                  # partition descriptor
                part_start, part_len = struct.unpack_from("<II", blob, 188)
            elif ident == 6:                                # logical volume
                fsd_block = struct.unpack_from("<I", blob, 248 + 4)[0]
            elif ident == 8:
                break
        if part_start is None or fsd_block is None:
            raise LayoutError("UDF volume descriptors are incomplete")

        fsd = sector(part_start + fsd_block)
        root_block = struct.unpack_from("<I", fsd, 400 + 4)[0]

        out: List[IsoFile] = []

        def read_fe(block: int) -> Tuple[int, int, List[Tuple[int, int]], int]:
            blob = sector(part_start + block)
            ident = struct.unpack_from("<H", blob, 0)[0]
            if ident not in (261, 266):
                raise LayoutError("expected a file entry at block %d" % block)
            base = 16
            ftype = blob[base + 11]
            flags = struct.unpack_from("<H", blob, base + 18)[0]
            head = 176 if ident == 261 else 216
            info_len = struct.unpack_from("<Q", blob, base + 40)[0]
            l_ea, l_ad = struct.unpack_from("<II", blob, head - 8)
            ads: List[Tuple[int, int]] = []
            if flags & 7 == 0:                              # short_ad
                for o in range(head + l_ea, head + l_ea + l_ad, 8):
                    ln, pos = struct.unpack_from("<II", blob, o)
                    if ln & 0x3FFFFFFF == 0:
                        continue
                    ads.append(((part_start + pos) * SECTOR, ln & 0x3FFFFFFF))
            elif flags & 7 == 1:                            # long_ad
                for o in range(head + l_ea, head + l_ea + l_ad, 16):
                    ln, pos = struct.unpack_from("<II", blob, o)
                    if ln & 0x3FFFFFFF == 0:
                        continue
                    ads.append(((part_start + pos) * SECTOR, ln & 0x3FFFFFFF))
            return ftype, info_len, ads, flags

        def walk(block: int, prefix: str, depth: int = 0) -> None:
            if depth > 64:
                return
            ftype, info_len, ads, _ = read_fe(block)
            stream = bytearray()
            for off, ln in ads:
                f.seek(off)
                stream += f.read(ln)
            pos = 0
            while pos + 38 <= len(stream):
                if struct.unpack_from("<H", stream, pos)[0] != 257:
                    pos = ((pos // SECTOR) + 1) * SECTOR
                    continue
                chars = stream[pos + 18]
                l_fi = stream[pos + 19]
                child = struct.unpack_from("<I", stream, pos + 20 + 4)[0]
                l_iu = struct.unpack_from("<H", stream, pos + 36)[0]
                name_raw = bytes(stream[pos + 38 + l_iu:pos + 38 + l_iu + l_fi])
                pos += align4(38 + l_iu + l_fi)
                if chars & 0x08:                            # parent entry
                    continue
                if name_raw[:1] == b"\x10":
                    name = name_raw[1:].decode("utf-16-be", "replace")
                elif name_raw[:1] == b"\x08":
                    name = name_raw[1:].decode("latin-1")
                else:
                    name = name_raw.decode("latin-1", "replace")
                full = prefix + "/" + name
                if chars & 0x02:
                    walk(child, full, depth + 1)
                else:
                    ctype, clen, cads, _ = read_fe(child)
                    if clen > 0 and cads:
                        out.append(IsoFile(path=full, lba=cads[0][0] // SECTOR,
                                           size=clen,
                                           padded=sum(align_up(l) for _, l in cads)))
                        out[-1].extents = cads              # type: ignore[attr-defined]

        walk(root_block, "")
    out.sort(key=lambda x: x.lba)
    first = out[0].lba * SECTOR if out else 0
    return out, size, first


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bdxl_layerguard.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Lay out a backup so only zero-filled buffers sit on a "
                    "BDXL's layer transitions.",
        epilog="examples:\n"
               "  %(prog)s build-udf ~/Pictures --buffer 512M -o backup.iso\n"
               "  %(prog)s verify backup.iso --buffer 512M\n"
               "  %(prog)s verify /dev/sr0 --raw --buffer 512M\n"
               "\nburn the image with ImgBurn in 'Write image file to disc'\n"
               "mode (never Build mode), SAO/DAO, single closed session.\n",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def geo_args(q):
        q.add_argument("--disc", choices=sorted(DISC_PRESETS), default=DEFAULT_DISC,
                       help="disc type (default %(default)s = 100 GB BDXL, "
                            "3 x 32,794,804,224 B)")
        q.add_argument("--layers", type=int, help="override the layer count")
        q.add_argument("--layer-bytes", help="override the bytes per layer")

    a = sub.add_parser("plan", help="scan a folder and build the staging tree")
    a.add_argument("source", help="folder containing the data to back up")
    a.add_argument("--buffer", required=True,
                   help="size of EACH zero-filled buffer file, e.g. 512M or 1G")
    a.add_argument("-o", "--out", default="./bdxl_staging",
                   help="staging folder to create (default %(default)s)")
    geo_args(a)
    a.add_argument("--bias", type=float, default=0.5,
                   help="fraction of the buffer placed before the transition "
                        "(default 0.5 = centred)")
    a.add_argument("--metadata-reserve", metavar="SIZE|@ISO",
                   help="bytes of filesystem metadata to assume in front of the "
                        "file data. Omit to estimate. Pass @some.iso to measure "
                        "it from a previously built image, which makes the "
                        "second pass exact")
    a.add_argument("--udf", action="store_true",
                   help="you will build the image with a UDF-capable tool "
                        "(NOT xorriso - it has no UDF writer). Enlarges the "
                        "metadata estimate accordingly; UDF placement is less "
                        "predictable, so use a bigger buffer and verify --raw")
    a.add_argument("--no-joliet", action="store_true",
                   help="you will build without -J (Windows-friendly names); "
                        "roughly halves the metadata estimate")
    a.add_argument("--link-mode", choices=("auto", "hardlink", "symlink", "copy"),
                   default="auto",
                   help="how to put source files in the staging tree "
                        "(default auto: hardlink, else symlink, else copy)")
    a.add_argument("--follow-symlinks", action="store_true",
                   help="follow symlinks in the source instead of skipping them")
    a.add_argument("--no-sparse", action="store_true",
                   help="write real zeros for the buffers instead of sparse files")
    a.add_argument("--no-manifest", action="store_true",
                   help="do not put a manifest on the disc")
    a.add_argument("--label", help="volume label for the generated burn command")
    a.add_argument("--iso", help="ISO filename for the generated burn command")
    a.add_argument("--dry-run", action="store_true",
                   help="print the layout and stop, write nothing")
    a.set_defaults(func=cmd_plan)

    c = sub.add_parser("build-udf",
                       help="write a ready-to-burn UDF image directly "
                            "(recommended: exact offsets, no 4 GiB limit, "
                            "tree preserved)")
    c.add_argument("source", help="folder containing the data to back up")
    c.add_argument("--buffer", required=True,
                   help="size of EACH zero-filled buffer file, e.g. 512M or 1G")
    c.add_argument("-o", "--out", default="./backup.iso",
                   help="image file to write (default %(default)s)")
    geo_args(c)
    c.add_argument("--bias", type=float, default=0.5,
                   help="fraction of the buffer placed before the transition "
                        "(default 0.5 = centred)")
    c.add_argument("--label", help="volume label")
    c.add_argument("--follow-symlinks", action="store_true",
                   help="follow symlinks in the source instead of skipping them")
    c.add_argument("--no-manifest", action="store_true",
                   help="do not put a manifest file on the disc")
    c.add_argument("--no-sparse", action="store_true",
                   help="write the buffers as real zeros rather than leaving "
                        "them as holes in the image file")
    c.add_argument("--dry-run", action="store_true",
                   help="print the layout and stop, write nothing")
    c.set_defaults(func=cmd_build_udf)

    b = sub.add_parser("verify", help="check a built ISO or a burned disc")
    b.add_argument("image", help="path to the .iso, or a device such as /dev/sr0")
    geo_args(b)
    b.add_argument("--buffer", help="expected buffer size, for margin warnings")
    b.add_argument("--raw", action="store_true",
                   help="skip the filesystem and just read the bytes around "
                        "each transition, checking they are all zero. Use this "
                        "for UDF images, hybrid images, and burned discs on "
                        "Windows (all reads are sector aligned)")
    b.add_argument("--window", help="how far either side of a transition --raw "
                                    "scans (default: --buffer, else 256M)")
    b.add_argument("--fs", choices=("auto", "udf", "iso9660"), default="auto",
                   help="which filesystem to read (default: detect)")
    b.set_defaults(func=cmd_verify)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    cfg = build_parser().parse_args(argv)
    try:
        return cfg.func(cfg)
    except LayoutError as exc:
        print("\nERROR: %s" % exc, file=sys.stderr)
        return 1
    except ValueError as exc:
        print("\nERROR: bad value: %s" % exc, file=sys.stderr)
        return 1
    except OSError as exc:
        print("\nERROR: %s" % exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
