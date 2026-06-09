# settings.py
import numpy as np

# Logging and output
# train_offsets_v1.py writes LOG_FILE and plots under OUTPUT_DIR.
OUTPUT_DIR = "outputs/latest"
LOG_FILE = "training_log.txt"

# Device Configuration
CPU_ONLY = False # Set to True to force CPU usage

# Environment Configuration
# Passed to FHSSQPSKEnv(**ENV_CONFIG)
ENV_CONFIG = {
    "Startfre": 3e6,
    "Endfre": 4e6,
    "Sub_interval": 50000,
    "Fs": 1e7,
    "Baud": 25000,
    "Hoprate": 100,             # Base hoprate
    "hoprate_min": 10.0,
    "hoprate_max": 1000.0,
    "enable_reactive": False,   # Reactive Jammer
    "enable_sweep": True,       # Indiscriminate Jammer (Swipe/Comb)
    "enable_rayleigh": True,    # Fading Channel
    "debug_plot_psd": False,
    "debug_log_hops": False,
    "use_pregen": True,
    "pregen_steps": 44          # Align with 4.4s cycle (0.1s step)
}

# Jammer Configuration
JAMMER_CONFIG = {
    # Global Jamming Mode: 'sweep', 'comb', or 'both'
    "mode": "comb",
    
    # Sweep Jamming Configuration
    "sweep": {
        "step": 50000,       # Frequency step for sweep
        "power": 0.8,        # Jamming power
        "dwell_time": 0.004, # Dwell time per step
        "bandwidth": 50000.0,# Noise bandwidth
    },
    
    # Comb Jamming Configuration
    "comb": {
        # Note: Frequency selection is HARDCODED in jammers.py to 8 fixed points (centers of 50kHz channels).
        "power": 0.8,        # Total power or per-tone power factor
        "bandwidth": 50000.0,# Noise bandwidth per tone
        # "frequencies": [] # Removed as it is now hardcoded in logic
    },
    
    # Reactive Jamming Configuration
    "reactive": {
        "speed": 160.0,
        "power": 0.8,
        "bandwidth": 50000.0,
    }
}

# SAC Agent Configuration
SAC_CONFIG = {
    "actor_lr": 1e-5,
    "critic_lr": 1e-4,
    "alpha_lr": 1e-4,
    "tau": 0.005,
    "gamma": 0.99,
    "target_entropy_ratio": 0.9,
}

# EAS-style local neighborhood search SAC variant
EAS_LOCAL_CONFIG = {
    "search_radius": 1,
    "distill_coef": 0.1,
    "search_eval": "min_q",
    "teacher_from_replay": True,
    "log_search_stats": True,
    "eas_replay_capacity": 12000,
    "eas_batch_size": 256,
    "filter_teacher_on_update": True,
    "teacher_compare_mode": "min_q",
}

# Replay Buffer Configuration
BUFFER_CONFIG = {
    "capacity": 12000,
    "batch_size": 256,
}

# Training Loop Configuration
TRAIN_CONFIG = {
    "steps_per_episode": 2000,       # Total environment steps per episode
    "min_buffer_before_train": 1200, # Warmup replay entries; each env step adds 10 entries
    "update_iters_per_step": 1,      # Gradient updates per environment step
    "fixed_hoprate": 200.0,          # Fixed hopping rate for training
}

# Reward Calculation Configuration
# Matches FHSSQPSKEnv.step(): base_reward - BER * ber_penalty - hoprate * hoprate_penalty
REWARD_CONFIG = {
    "base_reward": 0.5,
    "ber_penalty": 1.0,
    "hoprate_penalty": 0,
}
