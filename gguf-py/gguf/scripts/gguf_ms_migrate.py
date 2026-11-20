#!/usr/bin/env python3
# Rewrites an MS (Mad Shift) Gen 1 GGUF as a Gen 2 file: a new file, the input is only read.
#
# Gen 1 files (ms-llama.cpp release 1, ms-v1.0) store the MS block types MS_C1..MS_C6 as the ggml type ids 43..48 and
# general.file_type 42. Upstream ggml later gave 43 to F8_E4M3 (and file type 42 to MOSTLY_F8_E4M3), so MS moved to its
# reserved block: MS_C1..MS_C6 = 201..206, file type MOSTLY_MS = 200. The blocks are the same, so only the header changes:
# each type id is a uint32 in the tensor info and file_type a uint32 value, both rewritten in place in a copy of the file.
# The tensor data and every offset stay byte-identical; the output is checked against the input before it is kept.
#
# usage: python gguf-py/gguf/scripts/gguf_ms_migrate.py <gen1.gguf> <gen2.gguf>
#        python gguf-py/gguf/scripts/gguf_ms_migrate.py --check <file.gguf> [...]
from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
import sys
from pathlib import Path

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np  # noqa: E402

from gguf import GGUFReader  # noqa: E402
from gguf.constants import (  # noqa: E402
    GGML_QUANT_SIZES,
    GGMLQuantizationType,
    GGUFValueType,
    LlamaFileType,
    MS_GEN1_FILE_TYPE,
    MS_GEN1_TYPE_IDS,
    MS_GEN2_TYPE_IDS,
    ms_is_metadata_key,
)

logger = logging.getLogger("gguf-ms-migrate")

# Gen 1 id -> Gen 2 id, and the (block size, type size) of a Gen 1 id: the same blocks as Gen 2
GEN1_TO_GEN2 = {g1: g2 for g1, g2 in zip(MS_GEN1_TYPE_IDS, MS_GEN2_TYPE_IDS)}
GEN1_SIZES   = {g1: GGML_QUANT_SIZES[GGMLQuantizationType(g2)] for g1, g2 in GEN1_TO_GEN2.items()}


class HeaderReader(GGUFReader):
    # keeps the raw tensor info: this gguf-py reads the Gen 1 ids as other types (43 = F8_E4M3) or not at all (44..48)
    def _build_tensors(self, start_offs: int, fields: list) -> None:
        self.tensor_fields = fields
        self.tensors = []


def byte_offset(reader: GGUFReader, view: np.ndarray) -> int:
    # file offset of a view into the reader's memmap
    return view.__array_interface__['data'][0] - reader.data.__array_interface__['data'][0]


class Header:
    def __init__(self, path: Path, mode: str = 'r'):
        self.reader = HeaderReader(path, mode)
        r = self.reader
        self.is_ms   = any(ms_is_metadata_key(k) for k in r.fields)
        self.names   = [str(bytes(f.parts[1]), encoding='utf-8') for f in r.tensor_fields]
        self.types   = [int(f.parts[4][0]) for f in r.tensor_fields]
        self.offsets = [int(f.parts[5][0]) for f in r.tensor_fields]
        self.n_elems = []
        for f in r.tensor_fields:
            n = 1
            for d in f.parts[3].tolist():
                n *= int(d)
            self.n_elems.append(n)
        ft = r.fields.get('general.file_type')
        self.file_type = int(ft.parts[-1][0]) if ft is not None and ft.types == [GGUFValueType.UINT32] else None
        n1 = sum(t in MS_GEN1_TYPE_IDS for t in self.types)
        n2 = sum(t in MS_GEN2_TYPE_IDS for t in self.types)
        self.gen = 0 if not self.is_ms else (1 if n1 > 0 and n2 == 0 else 2 if n2 > 0 and n1 == 0 else -1)

    def describe(self) -> str:
        if not self.is_ms:
            return "not an MS file (no MS metadata)"
        return {1: "MS Gen 1 (type ids 43..48)", 2: "MS Gen 2 (type ids 201..206)"}.get(self.gen, "MS file with no MS types or with both id ranges")

    def check_gen1_layout(self) -> None:
        # with the Gen 1 sizes, the tensor data must be laid out as gguf.cpp expects: back to back, aligned, inside the file
        r = self.reader
        expected = 0
        end = 0
        for name, t, offs, n in zip(self.names, self.types, self.offsets, self.n_elems):
            bs, ts = GEN1_SIZES[t] if t in GEN1_SIZES else GGML_QUANT_SIZES[GGMLQuantizationType(t)]
            if n % bs != 0:
                raise ValueError(f"tensor {name!r}: {n} elements, not a multiple of the block size {bs} of type {t}")
            if offs != expected:
                raise ValueError(f"tensor {name!r}: offset {offs}, expected {expected} with the Gen 1 type sizes")
            end = offs + n // bs * ts
            expected = (end + r.alignment - 1) // r.alignment * r.alignment
        if r.data_offset + end > r.data.nbytes:
            raise ValueError(f"tensor data ends at {r.data_offset + end}, past the file size {r.data.nbytes}")

    def close(self) -> None:
        # drops the memmap: a mapped file cannot be renamed or replaced on Windows
        self.reader = None
        gc.collect()


def patch(path: Path) -> dict[int, tuple[int, int]]:
    # rewrites the ids in place; returns {file offset: (old byte, new byte)} of every byte that changes
    r = HeaderReader(path, 'r+')
    changes: dict[int, tuple[int, int]] = {}

    def set_u32(view: np.ndarray, new: int) -> None:
        offs = byte_offset(r, view)
        old_bytes = bytes(r.data[offs:offs + 4])
        view[0] = new
        new_bytes = bytes(r.data[offs:offs + 4])
        for i, (a, b) in enumerate(zip(old_bytes, new_bytes)):
            if a != b:
                changes[offs + i] = (a, b)

    for f in r.tensor_fields:
        t = int(f.parts[4][0])
        if t in GEN1_TO_GEN2:
            set_u32(f.parts[4], GEN1_TO_GEN2[t])
    ft = r.fields.get('general.file_type')
    if ft is not None and ft.types == [GGUFValueType.UINT32] and int(ft.parts[-1][0]) == MS_GEN1_FILE_TYPE:
        set_u32(ft.parts[-1], int(LlamaFileType.MOSTLY_MS))
    r.data.flush()
    return changes  # the memmap is released with r


def verify(src: Path, dst: Path, changes: dict[int, tuple[int, int]], data_offset: int) -> None:
    # the two files may differ only in the patched header bytes
    if src.stat().st_size != dst.stat().st_size:
        raise ValueError("output size differs from the input")
    if changes and max(changes) >= data_offset:
        raise ValueError("a patched byte lies in the tensor data")
    chunk = 64 << 20
    found: dict[int, tuple[int, int]] = {}
    with open(src, 'rb') as fa, open(dst, 'rb') as fb:
        pos = 0
        while True:
            a = fa.read(chunk)
            b = fb.read(chunk)
            if not a and not b:
                break
            if a != b:
                if len(a) != len(b):
                    raise ValueError("short read")
                av = np.frombuffer(a, dtype=np.uint8)
                bv = np.frombuffer(b, dtype=np.uint8)
                for i in np.nonzero(av != bv)[0].tolist():
                    found[pos + i] = (int(av[i]), int(bv[i]))
                    if len(found) > len(changes):
                        raise ValueError(f"unexpected difference at byte {pos + i}")
            pos += len(a)
    if found != changes:
        raise ValueError("the differing bytes are not the patched ids")

    # the output must read as a normal Gen 2 file with this gguf-py
    out = GGUFReader(dst, 'r')
    h_src = Header(src, 'r')
    try:
        if len(out.tensors) != len(h_src.names):
            raise ValueError("tensor count differs")
        for t, name, t_src, offs in zip(out.tensors, h_src.names, h_src.types, h_src.offsets):
            want = GEN1_TO_GEN2.get(t_src, t_src)
            if t.name != name or int(t.tensor_type) != want or t.data_offset != out.data_offset + offs:
                raise ValueError(f"tensor {name!r} reads back wrong")
    finally:
        h_src.close()


def migrate(src: Path, dst: Path, do_verify: bool) -> None:
    if dst.exists():
        raise SystemExit(f"{dst} exists, refusing to overwrite it")
    if src.resolve() == dst.resolve():
        raise SystemExit("the output must be a new file")

    h = Header(src, 'r')
    desc = h.describe()
    if h.gen != 1:
        h.close()
        raise SystemExit(f"{src}: {desc}, nothing to migrate")
    h.check_gen1_layout()
    counts: dict[str, int] = {}
    for t in h.types:
        if t in GEN1_TO_GEN2:
            name = GGMLQuantizationType(GEN1_TO_GEN2[t]).name
            counts[name] = counts.get(name, 0) + 1
    file_type = h.file_type
    data_offset = h.reader.data_offset
    h.close()

    part = dst.with_name(dst.name + '.part')
    logger.info(f"{src}: {desc}; copying to {part}")
    shutil.copyfile(src, part)
    try:
        changes = patch(part)
        gc.collect()
        if do_verify:
            logger.info("verifying: only the header ids may differ, the copy must read back as Gen 2")
            verify(src, part, changes, data_offset)
            gc.collect()
        os.replace(part, dst)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    ft = f", general.file_type {file_type} -> {int(LlamaFileType.MOSTLY_MS)}" if file_type == MS_GEN1_FILE_TYPE else ""
    logger.info(f"wrote {dst}: {', '.join(f'{n} x{c}' for n, c in sorted(counts.items()))} moved to 201..206{ft}; "
                f"{len(changes)} header bytes changed, tensor data unchanged")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite an MS (Mad Shift) Gen 1 GGUF (type ids 43..48) as Gen 2 (201..206) in a new file")
    parser.add_argument("--check", action="store_true", help="only print the MS generation of each file")
    parser.add_argument("--no-verify", action="store_true", help="skip the byte comparison of the output with the input")
    parser.add_argument("--verbose", action="store_true", help="more output")
    parser.add_argument("files", nargs="+", help="<gen1.gguf> <gen2.gguf>, or the files to check with --check")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    if args.check:
        for f in args.files:
            h = Header(Path(f), 'r')
            print(f"{f}: {h.describe()}")
            h.close()
        return
    if len(args.files) != 2:
        parser.error("give the Gen 1 input and the new output file")
    migrate(Path(args.files[0]), Path(args.files[1]), not args.no_verify)


if __name__ == '__main__':
    main()
