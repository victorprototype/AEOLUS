"""FermiNet config for the hydrogen atom (ground state).

Run (paths can be overridden on the command line):
    ferminet --config configs/hydrogen.py --config.log.save_path runs/h_run
"""
from ferminet import base_config
from ferminet.utils import system


def get_config():
    cfg = base_config.default()
    cfg.system.electrons = (1, 0)  # 1 alpha, 0 beta
    cfg.system.molecule = [system.Atom('H', (0, 0, 0))]
    cfg.batch_size = 256            # sized for CPU; raise on GPU
    cfg.pretrain.iterations = 100
    cfg.optim.iterations = 20000
    cfg.log.save_path = 'runs/h_run'
    return cfg
