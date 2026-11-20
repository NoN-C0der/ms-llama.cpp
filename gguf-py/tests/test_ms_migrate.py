#!/usr/bin/env python3
# MS (Mad Shift) Gen 1 -> Gen 2 migration (gguf/scripts/gguf_ms_migrate.py) on a tiny synthetic GGUF.
# pytest gguf-py/tests/test_ms_migrate.py, or: python gguf-py/tests/test_ms_migrate.py --write-dir DIR (writes gen1.gguf and
# gen2.gguf for a check of the C++ loader: a Gen 1 file must fail with "MS Gen 1 file", the Gen 2 file must load).
from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

import gguf  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType, LlamaFileType, MS_GEN1_TYPE_IDS, MS_GEN2_TYPE_IDS  # noqa: E402
from gguf.scripts import gguf_ms_migrate as mig  # noqa: E402


def write_gen2(path: Path) -> None:
    # an MS-like file: MS metadata, one tensor of each MS type (two rows of 512 weights) and an F32 tensor
    rng = np.random.default_rng(1)
    w = gguf.GGUFWriter(path, 'llama')
    w.add_string('msffn.code', '1mad')
    w.add_string('msffn.quant_name', 'MS-test')
    w.add_file_type(int(LlamaFileType.MOSTLY_MS))
    tensors = []
    for t in MS_GEN2_TYPE_IDS:
        qt = GGMLQuantizationType(t)
        bs, ts = GGML_QUANT_SIZES[qt]
        data = rng.integers(0, 256, size=(2, 512 // bs * ts), dtype=np.uint8)
        tensors.append((f'ms.{qt.name.lower()}', data, qt))
    tensors.append(('norm', rng.standard_normal(512).astype(np.float32), GGMLQuantizationType.F32))
    for name, data, qt in tensors:
        w.add_tensor(name, data, raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def make_gen1(src: Path, dst: Path) -> None:
    # the inverse of the migration: ids 201..206 -> 43..48, file type 200 -> 42
    shutil.copyfile(src, dst)
    r = mig.HeaderReader(dst, 'r+')
    for f in r.tensor_fields:
        t = int(f.parts[4][0])
        if t in MS_GEN2_TYPE_IDS:
            f.parts[4][0] = t - MS_GEN2_TYPE_IDS[0] + MS_GEN1_TYPE_IDS[0]
    ft = r.fields['general.file_type']
    ft.parts[-1][0] = 42
    r.data.flush()
    del r, f, ft
    gc.collect()


def test_reader_rejects_gen1(tmp_path):
    import pytest
    write_gen2(tmp_path / 'gen2.gguf')
    make_gen1(tmp_path / 'gen2.gguf', tmp_path / 'gen1.gguf')
    with pytest.raises(ValueError, match='MS Gen 1 file'):
        gguf.GGUFReader(tmp_path / 'gen1.gguf')
    r = gguf.GGUFReader(tmp_path / 'gen2.gguf')
    assert [int(t.tensor_type) for t in r.tensors[:6]] == list(MS_GEN2_TYPE_IDS)


def test_migrate_round_trip(tmp_path):
    write_gen2(tmp_path / 'gen2.gguf')
    make_gen1(tmp_path / 'gen2.gguf', tmp_path / 'gen1.gguf')
    assert mig.Header(tmp_path / 'gen1.gguf').gen == 1
    mig.migrate(tmp_path / 'gen1.gguf', tmp_path / 'out.gguf', do_verify=True)
    gc.collect()
    # the migrated file is byte for byte the Gen 2 file
    assert (tmp_path / 'out.gguf').read_bytes() == (tmp_path / 'gen2.gguf').read_bytes()
    assert not (tmp_path / 'out.gguf.part').exists()


def test_migrate_refuses(tmp_path):
    import pytest
    write_gen2(tmp_path / 'gen2.gguf')
    make_gen1(tmp_path / 'gen2.gguf', tmp_path / 'gen1.gguf')
    with pytest.raises(SystemExit):  # already Gen 2
        mig.migrate(tmp_path / 'gen2.gguf', tmp_path / 'x.gguf', do_verify=True)
    with pytest.raises(SystemExit):  # the output exists
        mig.migrate(tmp_path / 'gen1.gguf', tmp_path / 'gen2.gguf', do_verify=True)
    with pytest.raises(SystemExit):  # same file
        mig.migrate(tmp_path / 'gen1.gguf', tmp_path / 'gen1.gguf', do_verify=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write-dir', required=True, help='directory for gen1.gguf, gen2.gguf and the migrated migrated.gguf')
    a = ap.parse_args()
    d = Path(a.write_dir)
    d.mkdir(parents=True, exist_ok=True)
    for n in ('gen1.gguf', 'gen2.gguf', 'migrated.gguf'):
        (d / n).unlink(missing_ok=True)
    write_gen2(d / 'gen2.gguf')
    make_gen1(d / 'gen2.gguf', d / 'gen1.gguf')
    mig.migrate(d / 'gen1.gguf', d / 'migrated.gguf', do_verify=True)
    gc.collect()
    if (d / 'migrated.gguf').read_bytes() != (d / 'gen2.gguf').read_bytes():
        raise SystemExit('the migrated file differs from the Gen 2 file')
    print(f'wrote {d / "gen1.gguf"}, {d / "gen2.gguf"}; the migration of gen1 equals gen2')


if __name__ == '__main__':
    main()
