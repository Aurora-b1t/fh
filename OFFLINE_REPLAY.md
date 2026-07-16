# Offline replay workflow

The offset SAC and MBPO-SAC entry points load real replay transitions before
their first training update. The default file is configured in `settings.py`:

```text
outputs/offline_replay/replay_50000_random_hoprate.npz
```

Generate the default 50,000 block-level transitions with random hoprates and
random offset actions:

```bash
python generate_offline_replay.py
```

Generate the same number with a fixed hoprate:

```bash
python generate_offline_replay.py --hoprate_mode fixed --fixed_hoprate 100 --output_path outputs/offline_replay/replay_50000_fixed_100.npz
```

The generator runs the real `FHSSQPSKEnv`. One environment step produces ten
block-level transitions, so 50,000 transitions require 5,000 environment steps.
The last environment step is truncated in the saved replay if the requested
count is not a multiple of ten.

Select a data file for either training scheme with the same option:

```bash
python train_offsets.py --offline_replay_path outputs/offline_replay/replay_50000_fixed_100.npz
python train_mbpo.py --offline_replay_path outputs/offline_replay/replay_50000_fixed_100.npz
```

The path can also be changed through `settings.OFFLINE_REPLAY_CONFIG`. The
loader validates the transition fields, observation shape, action range, block
count, and replay capacity. It warns when the stored environment configuration
differs from the current one, which makes it possible to keep multiple replay
files for different environment versions while still failing fast on tensor
shape or action-space incompatibility.

There is no replay warm-up gate anymore. The preloaded real buffer is used by
the first SAC update, and subsequent online transitions continue to enter the
same real buffer. MBPO trains its reward model from this real buffer and keeps
generated samples in its separate model buffer.
