import json

import numpy as np

from aeolus.dataset import block_error
from aeolus.export_web import export


def _fake_npz(path):
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(12, 3)).astype(np.float32)
    step = np.repeat([100, 200, 300], 4)
    np.savez(path, walker_positions=pos, walker_checkpoint_step=step)


def test_export_groups_walkers_by_checkpoint(tmp_path):
    npz, out = tmp_path / "d.npz", tmp_path / "d.json"
    _fake_npz(npz)
    info = export(str(npz), str(out))
    data = json.loads(out.read_text())
    assert info == {"checkpoints": 3, "walkers_per_checkpoint": 4}
    assert data["steps"] == [100, 200, 300]
    assert len(data["snaps"]) == 3 and len(data["snaps"][0]) == 4 and len(data["snaps"][0][0]) == 3


def test_block_error_is_small_for_constant_signal():
    assert block_error(np.ones(1000)) == 0.0
