"""Build the dataset file for a FermiNet run.

Reads the training run (``train_stats.csv`` plus ``qmcjax_ckpt_*.npz`` checkpoints) and,
optionally, an inference run that logged observables, and writes one ``.npz`` file.

Note: loading FermiNet checkpoints needs JAX installed, because the stored walker data are
pickled JAX arrays.

Usage:
    python -m aeolus.dataset --run runs/h_run --inf runs/h_inf --out data/hydrogen_v0.npz
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd


def block_error(x: np.ndarray, nblocks: int = 50) -> float:
    """Standard error of the mean from block averaging (robust to correlated samples)."""
    size = len(x) // nblocks
    if size < 1:
        raise ValueError(f"need at least {nblocks} samples, got {len(x)}")
    blocks = x[: nblocks * size].reshape(nblocks, size).mean(axis=1)
    return float(blocks.std(ddof=1) / np.sqrt(nblocks))


def load_walkers(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Return walker positions (N, 3) and the checkpoint step each one came from."""
    files = sorted(glob.glob(os.path.join(run_dir, "qmcjax_ckpt_*.npz")))
    if not files:
        raise FileNotFoundError(f"no qmcjax_ckpt_*.npz checkpoints in {run_dir}")
    pos, steps = [], []
    for f in files:
        ckpt = np.load(f, allow_pickle=True)
        p = np.asarray(ckpt["data"].item()["positions"]).reshape(-1, 3)
        pos.append(p)
        steps.append(np.full(len(p), int(ckpt["t"])))
    return np.concatenate(pos).astype(np.float32), np.concatenate(steps)


def load_observables(inf_dir: str, skip: int = 1000) -> dict[str, np.ndarray]:
    """Collect FermiNet observables from an inference run, if they were logged."""
    obs: dict[str, np.ndarray] = {}
    csv = os.path.join(inf_dir, "train_stats.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        if "s2" in df.columns:
            obs["s2"] = df["s2"].values[skip:]
        if {"mu_x", "mu_y", "mu_z"} <= set(df.columns):
            obs["dipole"] = df[["mu_x", "mu_y", "mu_z"]].values[skip:]

    # Density matrix: several arrays written back to back into one .npy file.
    # Untested against a real run; keep both views until we know which one is meaningful.
    dm = os.path.join(inf_dir, "density_matrix.npy")
    if os.path.exists(dm):
        mats = []
        with open(dm, "rb") as f:
            while True:
                try:
                    mats.append(np.load(f))
                except EOFError:
                    break
        if mats:
            obs["density_matrix_last"] = mats[-1]
            obs["density_matrix_mean"] = np.mean(mats, axis=0)
    return obs


def build(run_dir: str, inf_dir: str | None, out: str, burn: int = 5000) -> dict:
    stats = pd.read_csv(os.path.join(run_dir, "train_stats.csv"))
    d = stats.iloc[burn:]
    energy = d["energy"].values
    err = block_error(energy)

    pos, step_of = load_walkers(run_dir)
    obs = load_observables(inf_dir) if inf_dir else {}

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(
        out,
        atomic_number=1,
        nuclear_position=np.zeros(3),
        units=np.array("bohr"),
        exact_energy=-0.5,
        energy_step=d["step"].values,
        energy=energy,
        pmove=d["pmove"].values,
        energy_offset_from_exact=energy.mean() + 0.5,
        energy_stderr=err,
        walker_positions=pos,
        walker_checkpoint_step=step_of,
        mean_r=np.linalg.norm(pos, axis=1).mean(),
        **obs,
    )
    return {"offset": energy.mean() + 0.5, "stderr": err,
            "mean_r": float(np.linalg.norm(pos, axis=1).mean()), "observables": sorted(obs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="training run folder (train_stats.csv + checkpoints)")
    ap.add_argument("--inf", default=None, help="optional inference run folder with observables")
    ap.add_argument("--out", default="data/hydrogen_v0.npz")
    ap.add_argument("--burn", type=int, default=5000, help="steps to skip at the start of the energy trace")
    a = ap.parse_args()
    info = build(a.run, a.inf, a.out, a.burn)
    print(f"wrote {a.out}")
    print(f"energy offset {info['offset']:.2e} +- {info['stderr']:.2e} Eh | mean r {info['mean_r']:.3f} bohr "
          f"| observables: {info['observables'] or 'none yet'}")


if __name__ == "__main__":
    main()
