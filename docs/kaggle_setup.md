# Running FermiNet on Kaggle

Notes from getting a hydrogen run working in October 2026. Library versions move quickly, so treat the
specific version numbers as a record of what worked, not a guarantee.

## What worked

CPU only, Python 3.13, JAX 0.9.2. In a notebook with the accelerator set to **None** and Internet **on**:

```
!pip uninstall -y jax-cuda12-plugin jax-cuda12-pjrt
!pip install -U "jax>=0.9,<0.10"
```

Restart the session if you imported `jax` earlier, then verify:

```
!pip freeze | grep -i -E "^(jax|jaxlib|jax-cuda|numpy|scipy)"
import jax; print(jax.__version__, jax.devices())   # expect 0.9.x and a CpuDevice
```

Install FermiNet afterwards, so pip leaves JAX alone (FermiNet's `setup.py` does not pin it):

```
!git clone https://github.com/google-deepmind/ferminet.git
%cd ferminet
!pip install -e .
```

Then run the config from this repo (batch size 256 suits CPU, about 0.23 s per step):

```
!JAX_PLATFORMS=cpu python ferminet/main.py --config /path/to/configs/hydrogen.py
```

## What did not work

| Setup | Failure |
|---|---|
| Latest JAX (0.11.1) | `kfac_jax` calls `jax.device_put_replicated`, which was removed in JAX 0.10.0 |
| JAX 0.7.2 | `kfac_jax` asserts `jax.NamedSharding` on the step counter (needs the newer pmap behaviour) |
| 2x Tesla T4 | NCCL `ncclAllGather ... invalid argument` during the first gather. `CUDA_VISIBLE_DEVICES=0` avoids it |

The README of FermiNet only shows an old `jaxlib==0.1.57+cuda110` install line, which is not usable today.
`kfac_jax` and `folx` are installed from GitHub main, so a fresh install can pick up newer commits.

## Inference with observables

After training, freeze the network and measure observables. `iterations` is the final step number, so it
must exceed the step of the newest checkpoint (training resumes counting from the checkpoint):

```
!JAX_PLATFORMS=cpu python ferminet/main.py --config /path/to/configs/hydrogen.py \
  --config.optim.optimizer 'none' --config.optim.iterations 38023 \
  --config.log.restore_path runs/h_run --config.log.save_path runs/h_inf \
  --config.observables.s2 --config.observables.dipole --config.observables.density
```

Check that the log reports a restored checkpoint and that the first step is near the checkpoint's step.
The `s2` and dipole values are logged as extra columns in `train_stats.csv`; the density goes to
`density_matrix.npy`. For hydrogen, expect S² = 0.75 and a dipole of about 0.

## Saving your work

Files in `/kaggle/working` are lost when the session ends. Zip the run folder and download it, or use
`IPython.display.FileLink('file.zip')` from `/kaggle/working` for a direct link.
