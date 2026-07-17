"""Standalone configuration for the special hopping-pattern experiment."""

import os
import random

import numpy as np


CPU_ONLY = False
RANDOM_SEED = 42

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(TEST_DIR, "results")
CAPTURE_STEPS = [1, 75, 250]
CHANNEL_WIDTH = 50000.0


def set_random_seeds(seed=None):
    """Seed Python, NumPy, and PyTorch using the experiment seed."""
    if seed is None:
        seed = RANDOM_SEED

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# Each 100 ms block contains ten hops at a fixed 100 Hz hop rate. Block
# numbering in the experiment is one-based: block 1 uses ODD_BLOCK_HOPS.
ODD_BLOCK_HOPS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
EVEN_BLOCK_HOPS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
BLOCK_HOP_PATTERNS = [
    ODD_BLOCK_HOPS if block_number % 2 == 1 else EVEN_BLOCK_HOPS
    for block_number in range(1, 11)
]

COMB_PHASE_CHANNELS = [
    [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19],
]

EXPECTED_OFFSETS = [
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
]


ENV_CONFIG = {
    "Startfre": 3e6,
    "Endfre": 4e6,
    "Sub_interval": CHANNEL_WIDTH,
    "Fs": 1e7,
    "Baud": 25000,
    "Hoprate": 100,
    "hoprate_min": 10.0,
    "hoprate_max": 1000.0,
    "enable_reactive": False,
    "enable_sweep": True,
    "enable_rayleigh": False,
    "debug_plot_psd": False,
    "debug_log_hops": False,
    "reset_mseq_each_step": True,
    "use_pregen": True,
    "pregen_steps": 44,
    "noise_std": 0.1,
    "signal_power": 0.0025,
}


JAMMER_CONFIG = {
    "mode": "comb",
    "sweep": {
        "step": CHANNEL_WIDTH,
        "power": 0.8,
        "dwell_time": 0.004,
        "bandwidth": CHANNEL_WIDTH,
    },
    "comb": {
        "power": 0.8,
        "bandwidth": CHANNEL_WIDTH,
        "sub_interval": CHANNEL_WIDTH,
        "phase_channels": COMB_PHASE_CHANNELS,
    },
    "reactive": {
        "power": 1.5,
        "bandwidth": CHANNEL_WIDTH,
        "p_fa": 0.1,
        "detection_time": 0.0005,
    },
}


# Keep the current baseline SAC and online-training parameters unchanged.
SAC_CONFIG = {
    "actor_lr": 1e-5,
    "critic_lr": 1e-4,
    "alpha_lr": 1e-4,
    "tau": 0.005,
    "gamma": 0.95,
    "target_entropy_ratio": 0.1,
}

BUFFER_CONFIG = {
    "capacity": 50000,
    "batch_size": 256,
}

TRAIN_CONFIG = {
    "steps_per_episode": 250,
    "update_iters_per_step": 10,
    "fixed_hoprate": 100.0,
}

REWARD_CONFIG = {
    "base_reward": 4.0,
    "ber_penalty": 32.0,
    "hoprate_penalty": 0,
}
