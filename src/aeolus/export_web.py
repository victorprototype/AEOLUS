"""Convert the dataset ``.npz`` into the JSON file the web viewer loads.

Usage:
    python -m aeolus.export_web --npz data/hydrogen_v0.npz --out web/data/hydrogen_v0.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def export(npz_path: str, out_path: str, decimals: int = 3) -> dict:
    z = np.load(npz_path)
    pos = z["walker_positions"]
    step = z["walker_checkpoint_step"]
    steps = [int(s) for s in np.unique(step)]
    snaps = [np.round(pos[step == s], decimals).tolist() for s in steps]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"steps": steps, "snaps": snaps}, f, separators=(",", ":"))
    return {"checkpoints": len(steps), "walkers_per_checkpoint": len(snaps[0])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="data/hydrogen_v0.npz")
    ap.add_argument("--out", default="web/data/hydrogen_v0.json")
    a = ap.parse_args()
    info = export(a.npz, a.out)
    print(f"wrote {a.out}: {info['checkpoints']} checkpoints x {info['walkers_per_checkpoint']} walkers")


if __name__ == "__main__":
    main()
