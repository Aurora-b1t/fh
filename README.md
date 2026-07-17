# 强化学习跳频抗干扰系统（进行中）

> **状态：项目进行中，尚未完成。** 环境与多个算法入口可运行，但训练效果验证、算法稳定性、部分模块联调仍在推进。下列文档描述的是当前代码快照的真实状态，而非最终目标。

本项目是一个面向跳频扩频（FHSS, Frequency Hopping Spread Spectrum）抗干扰研究的 Python 实验环境。当前代码快照包含：

- **FHSS/QPSK 通信仿真环境**：QPSK 收发链路、跳频信道、PSD waterfall 观测、Rayleigh 衰落、扫频/梳状/反应式干扰机，支持预生成加速。
- **baseline 离散 SAC 训练**：agent 在每个环境 step 内自回归选择 10 个跳频 offset，actor/critic 各自维护独立 embedding。
- **MBPO 奖励模型增强 SAC**：训练一个 ensemble 奖励预测模型，生成合成 replay 辅助 SAC 更新（详见 [MBPO_MODULE.md](MBPO_MODULE.md)）。
- **Noisy Binary Search 跳速阈值搜索**：MWU-based noisy binary search 寻找反应式干扰机的跳速跟踪/失效边界，含 derivative 变体。
- **hoprate sweep 评估**：确定性网格遍历所有候选 hoprate，作为 NBS 搜索的对照基线。
- **离线 replay 生成与加载**：生成/复用真实环境 replay 数据，供 offset/MBPO 训练冷启动（详见 [OFFLINE_REPLAY.md](OFFLINE_REPLAY.md)）。
- **special hopping pattern 隔离测试**：独立子目录环境/干扰机/训练脚本，用于验证固定 comb 干扰下的可解 offset 模式。

项目根目录采用平铺结构，入口清晰、文件职责清晰。每个训练/搜索脚本自带默认输出目录，互不覆盖。

## 项目文件说明

### 根目录

| 文件 | 作用 |
| --- | --- |
| [settings.py](settings.py) | 统一配置环境参数、干扰机参数、SAC/MBPO/NBS 超参数、训练循环和奖励系数。 |
| [fh_env.py](fh_env.py) | FHSS/QPSK Gymnasium 环境：预生成加速、跳频序列、QPSK 收发、干扰/衰落叠加、BER 与 reward 计算、PSD waterfall 观测。 |
| [jammers.py](jammers.py) | 干扰机实现：快速带限噪声源、基于能量检测的反应式干扰机、扫频/梳状宽带干扰机。 |
| [SAC.py](SAC.py) | baseline 离散动作 SAC：ReplayBuffer、EASReplayBuffer、CNN policy/Q 网络、温度系数自适应、软更新。actor 独立维护 embedding。 |
| [train_offsets.py](train_offsets.py) | baseline 训练入口：构建环境、SAC agent、replay buffer、训练循环、日志和曲线输出。 |
| [train_mbpo.py](train_mbpo.py) | MBPO 奖励模型 + SAC 训练入口：真实环境交互 + 合成 replay 混合更新。 |
| [train_speed.py](train_speed.py) | Noisy Binary Search 跳速阈值搜索入口（随机 offset + 反应式干扰机）。 |
| [train_speed_derivative.py](train_speed_derivative.py) | derivative-based NBS 变体入口，用 BER-hoprate 导数指标做方向决策。 |
| [train_speed_sweep.py](train_speed_sweep.py) | hoprate 网格扫描评估入口，NBS 搜索的确定性对照。 |
| [noisy_binary_search.py](noisy_binary_search.py) | MWU-based noisy binary search 算法实现，参考 Dereniowski et al. STACS 2025。 |
| [noisy_binary_search_derivative.py](noisy_binary_search_derivative.py) | derivative-based NBS 算法实现。 |
| [offline_replay.py](offline_replay.py) | 离线 replay 序列化与 transition 拆分工具，供 generator 与训练脚本共用。 |
| [generate_offline_replay.py](generate_offline_replay.py) | 离线真实 replay 生成脚本，输出 `.npz` 文件。 |

### `r_predict_model/` 子包

| 文件 | 作用 |
| --- | --- |
| [r_predict_model/model.py](r_predict_model/model.py) | ensemble 奖励预测模型：标准化器、ensemble 全连接层、概率训练、holdout 验证、elite 选择。 |
| [r_predict_model/mbpo_adapter.py](r_predict_model/mbpo_adapter.py) | SAC replay 与奖励模型输入格式适配层：flatten 特征、提取标签、rollout 合成样本。 |
| [r_predict_model/replay_memory.py](r_predict_model/replay_memory.py) | 通用 MBPO 模板中的环形 replay memory。 |
| [r_predict_model/main.py](r_predict_model/main.py) | 通用 MBPO 训练模板，保留为参考实现，非当前主入口。 |
| [r_predict_model/__init__.py](r_predict_model/__init__.py) | 暴露 `EnsembleDynamicsModel`。 |

### `special_hopping_test/` 子目录

独立隔离测试套件：固定 comb 干扰下两组交替信道相位 + 特殊跳频模式，用于验证 agent 能否学到可解的 10-offset 模式。使用 20 信道、不同 reward 配置，与主目录环境互不依赖。

| 文件 | 作用 |
| --- | --- |
| [special_hopping_test/SAC_test.py](special_hopping_test/SAC_test.py) | 隔离测试专用 SAC 实现（与根目录 `SAC.py` 结构一致）。 |
| [special_hopping_test/fh_env_test.py](special_hopping_test/fh_env_test.py) | 隔离测试专用 FHSS 环境。 |
| [special_hopping_test/jammers_test.py](special_hopping_test/jammers_test.py) | 隔离测试专用干扰机实现。 |
| [special_hopping_test/test_settings.py](special_hopping_test/test_settings.py) | 隔离测试专用配置。 |
| [special_hopping_test/train_test.py](special_hopping_test/train_test.py) | 隔离测试训练入口。 |
| [special_hopping_test/validate_pattern.py](special_hopping_test/validate_pattern.py) | 不跑 RF 仿真的纯逻辑校验：验证特殊跳频/comb 构造的碰撞数与期望 offset。 |
| [special_hopping_test/validate_psd.py](special_hopping_test/validate_psd.py) | 用真实 RF 配置生成并校验 comb 干扰 PSD。 |

### 其它

| 文件 | 作用 |
| --- | --- |
| [example/main.py](example/main.py) | 通用 SAC + MBPO 参考训练模板（非 FHSS 入口）。 |
| [example/model.py](example/model.py)、[example/replay_memory.py](example/replay_memory.py) | 参考 MBPO 模型与 replay memory。 |
| `pdf/` | 参考文献 PDF：Dereniowski et al. STACS 2025、Urkowitz 能量检测。 |

历史文件说明：旧训练/启动脚本（如 `run.py`、`speedDQN.py`、`Jamming Strategy.md` 等）已不作为当前维护入口，如需查看可通过 git 历史找回。

## 运行环境

用户当前环境为：

```bash
D:\Anaconda\envs\rl_fhss\python.exe
```

代码中主要使用：

- `numpy` / `matplotlib` / `scipy` / `gymnasium` / `torch` / `commpy`

建议在项目根目录运行所有命令。以下示例均使用上述 Python 解释器，可替换为你环境中的等价路径。

## Baseline：离散 SAC offset 训练

查看参数：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets.py --help
```

短训练冒烟测试：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets.py --steps_per_episode 3 --batch_size 20 --offline_replay_path outputs/offline_replay/replay_50000_random_hoprate_v2.npz --output_dir outputs/smoke
```

默认训练（默认输出目录 `outputs/offsets/pre50000/comb/512_start`）：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets.py
```

强制 CPU：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets.py --cpu_only
```

指定输出目录：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets.py --output_dir outputs/baseline
```

## MBPO 奖励模型增强 SAC

[train_mbpo.py](train_mbpo.py) 在 baseline SAC 之外训练一个 ensemble 奖励预测模型（只预测一步 reward，不预测下一帧 PSD），用合成 replay 按 `real_ratio` 比例混入 SAC 更新。完整设计、数据流与配置说明见 [MBPO_MODULE.md](MBPO_MODULE.md)。

短冒烟测试：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_mbpo.py --steps_per_episode 5 --batch_size 20 --model_train_freq 2 --rollout_batch_size 20 --offline_replay_path outputs/offline_replay/replay_50000_random_hoprate_v2.npz --output_dir outputs/mbpo_smoke
```

默认训练（默认输出目录 `outputs/mbpo/comb/pre50000`）：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_mbpo.py
```

## 跳速阈值搜索（Noisy Binary Search）

[train_speed.py](train_speed.py) 使用 [noisy_binary_search.py](noisy_binary_search.py) 中的 MWU-based noisy binary search 算法，在反应式干扰机开启时搜索跳速阈值。每个环境 step 中：

1. NBS 给出一个待测试 hoprate。
2. 训练脚本随机生成 10 个 offset（不训练 SAC）。
3. 环境执行 10 个 block 并返回 BER。
4. NBS 根据当前 BER 与上一次 BER 的变化更新候选跳速权重分布。

运行示例（默认输出目录 `outputs/speed`）：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_speed.py --steps 60 --output_dir outputs/speed_test
```

自定义 NBS 参数：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_speed.py --nbs_p 0.15 --nbs_delta 0.03 --nbs_step 10 --steps 100 --output_dir outputs/speed_custom
```

### derivative NBS 变体

[train_speed_derivative.py](train_speed_derivative.py) 使用 [noisy_binary_search_derivative.py](noisy_binary_search_derivative.py) 的 derivative-based NBS 算法，用 BER-hoprate 导数指标代替单纯 BER 升降做方向决策：

```
metric = ΔBER_percent / Δhoprate_clamped
metric > threshold  → LEFT move（支持更小 hoprate）
metric ≤ threshold  → RIGHT move（支持更大 hoprate）
```

运行示例（默认输出目录 `outputs/speed_derivative/0.5ms/-0.005`）：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_speed_derivative.py --steps 60 --output_dir outputs/speed_test_derivative
D:\Anaconda\envs\rl_fhss\python.exe train_speed_derivative.py --derivative_threshold -0.005 --steps 100
```

### hoprate 网格扫描评估

[train_speed_sweep.py](train_speed_sweep.py) 是 [train_speed.py](train_speed.py) 的确定性网格对照：不使用 NBS 选择下一个 hoprate，而是按升序遍历每个候选 hoprate，记录 BER/reward 诊断。每个环境 step 内部执行 10 个 block，与训练脚本的环境 step 语义一致。

运行示例（默认输出目录 `outputs/speed_sweep/0.5ms`）：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_speed_sweep.py --output_dir outputs/speed_sweep
```

冒烟测试：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_speed_sweep.py --hoprate_max 20 --steps_per_hoprate 1
```

### 离线 replay 生成与加载

offset 与 MBPO 训练入口在首次梯度更新前加载真实 replay transitions，默认文件由 [OFFLINE_REPLAY.md](OFFLINE_REPLAY.md) 说明。生成默认 50,000 条 block-level transitions：

```bash
D:\Anaconda\envs\rl_fhss\python.exe generate_offline_replay.py
```

固定 hoprate 版本：

```bash
D:\Anaconda\envs\rl_fhss\python.exe generate_offline_replay.py --hoprate_mode fixed --fixed_hoprate 100 --output_path outputs/offline_replay/replay_50000_fixed_100_v2.npz
```

指定 replay 文件给训练入口：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets.py --offline_replay_path outputs/offline_replay/replay_50000_fixed_100_v2.npz
D:\Anaconda\envs\rl_fhss\python.exe train_mbpo.py   --offline_replay_path outputs/offline_replay/replay_50000_fixed_100_v2.npz
```

### special hopping pattern 隔离测试

在 `special_hopping_test/` 目录下独立运行（它有独立的 settings/env/jammers）：

```bash
cd special_hopping_test
D:\Anaconda\envs\rl_fhss\python.exe train_test.py --steps_per_episode 250 --output_dir results
```

纯逻辑校验（不跑 RF 仿真）：

```bash
cd special_hopping_test
D:\Anaconda\envs\rl_fhss\python.exe validate_pattern.py
```

PSD 校验：

```bash
cd special_hopping_test
D:\Anaconda\envs\rl_fhss\python.exe validate_psd.py
```

### NBS / 搜索主要输出

- `training_log.txt`：搜索日志。
- `hoprate.png`：实际测试 hoprate 与 NBS 估计轨迹。
- `ber.png`：每 step 平均 BER。
- `ber_vs_hoprate.png`：BER 与 hoprate 散点图。
- `nbs_weights.png`：最终候选 hoprate 权重分布。
- `nbs_distribution.npz`：候选集合、权重、测试 hoprate、BER 的 numpy 数据。
- `hoprate_sweep.csv` / `hoprate_sweep.npz`：sweep 评估的 CSV 和 numpy 数据。

相关参考：

- [Dereniowski 等 - 2025 - Noisy (Binary) Searching Simple, Fast and Correct.pdf](<pdf/Dereniowski 等 - 2025 - Noisy (Binary) Searching Simple, Fast and Correct.pdf>)
- [Energy_detection_of_unknown_deterministic_signals.pdf](pdf/Energy_detection_of_unknown_deterministic_signals.pdf)

## 输出目录

输出目录不再集中在 [settings.py](settings.py)，而是由各训练/搜索脚本自带默认 `--output_dir`，均可通过命令行覆盖：

| 脚本 | 默认 `--output_dir` |
| --- | --- |
| [train_offsets.py](train_offsets.py) | `outputs/offsets/pre50000/comb/512_start` |
| [train_mbpo.py](train_mbpo.py) | `outputs/mbpo/comb/pre50000` |
| [train_speed.py](train_speed.py) | `outputs/speed` |
| [train_speed_derivative.py](train_speed_derivative.py) | `outputs/speed_derivative/0.5ms/-0.005` |
| [train_speed_sweep.py](train_speed_sweep.py) | `outputs/speed_sweep/0.5ms` |
| [special_hopping_test/train_test.py](special_hopping_test/train_test.py) | `special_hopping_test/results` |

每个脚本还支持 `--log_file`（默认 `training_log.txt`，位于对应 `--output_dir` 内）。常见输出文件包括：

- `training_log.txt`：训练或搜索日志。
- `reward.png`：平均 step reward 曲线（offset/MBPO 训练）。
- `ber.png`：平均 step BER 曲线。
- `loss.png`：actor/critic loss 曲线（offset/MBPO 训练）。
- `model_reward.png`：奖励模型预测曲线（MBPO 训练）。
- `hoprate.png`、`ber_vs_hoprate.png`、`nbs_weights.png`：NBS 搜索诊断图。
- `hoprate_sweep.csv`、`hoprate_sweep.npz`：sweep 评估数据。
- PSD capture 图：指定 step 的观测与 10 个 block PSD（special hopping 测试）。

`outputs/` 已加入 [.gitignore](.gitignore)，训练产物默认不进入版本控制。

## 配置说明

主要配置集中在 [settings.py](settings.py)。

### 设备

- `CPU_ONLY`：是否强制使用 CPU。
- `set_random_seeds()`：统一设置 Python/NumPy/PyTorch 随机种子，并设置 `cudnn.deterministic=True, benchmark=False` 以保证可复现性。各训练入口不再单独覆盖此设置。

### `ENV_CONFIG`

传给 `FHSSQPSKEnv(**ENV_CONFIG)` 的环境参数，主要包括：

- `Startfre` / `Endfre`：FHSS 工作频段。
- `Sub_interval`：子信道间隔。
- `Fs`：采样率。
- `Baud`：码元率。
- `Hoprate`、`hoprate_min`、`hoprate_max`：基础跳速和跳速范围。
- `enable_reactive`：是否启用反应式干扰机。
- `enable_sweep`：是否启用扫频/梳状干扰机。
- `enable_rayleigh`：是否启用 Rayleigh 衰落。
- `use_pregen`、`pregen_steps`：是否使用预生成加速路径以及预生成 observation 周期长度。
- `noise_std`、`signal_power`：接收端噪声和反应式干扰检测所需信号功率参数。

### `JAMMER_CONFIG`

干扰机配置：

- `mode`：`sweep`、`comb` 或 `both`。
- `sweep`：扫频干扰的步进、功率、驻留时间、噪声带宽。
- `comb`：梳状干扰的功率和单 tone 带宽。当前 comb 频点选择在 [jammers.py](jammers.py) 中实现为两组交替的 8 个 50 kHz 对齐信道。
- `reactive`：反应式干扰机的功率、带宽、虚警概率、检测时长等。检测逻辑基于能量检测理论，按 1 ms slot 扫描/检测/压制。

### `SAC_CONFIG`

baseline SAC 超参数：`actor_lr`、`critic_lr`、`alpha_lr`、`tau`、`gamma`、`target_entropy_ratio`。

### `MBPO_CONFIG`

MBPO 奖励模型配置，详见 [MBPO_MODULE.md](MBPO_MODULE.md)。

### `BUFFER_CONFIG`

普通 replay buffer 配置：`capacity`、`batch_size`。

### `NBS_CONFIG`

Noisy Binary Search 配置：

- `p`：假设的噪声概率，需满足 `0 <= p < 0.5`。
- `delta`：收敛阈值；当最大权重 `>= 1 - delta` 时认为收敛。
- `hoprate_step`：候选 hoprate 离散步长，默认 10 Hz，与环境 `_apply_hoprate()` 的量化一致。

### `TRAIN_CONFIG`

训练循环配置：

- `steps_per_episode`：单次训练运行的环境 step 数。
- `update_iters_per_step`：每个环境 step 后的梯度更新次数。
- `fixed_hoprate`：offset 训练时使用的固定 hoprate。
- 离线 replay 在首次梯度更新前加载，文件路径由 `OFFLINE_REPLAY_CONFIG` 或 `--offline_replay_path` 配置。

### `REWARD_CONFIG`

训练脚本中逐 block reward 的计算参数：

```text
reward = base_reward - ber_penalty * BER - hoprate_penalty * hoprate
```

注意：环境 [fh_env.py](fh_env.py) 中的 step reward 当前为 `0.5 - mean_ber - hoprate_used * 0.0001`；训练脚本使用 `REWARD_CONFIG` 为每个 block 生成 replay reward。两者都是当前实验设计的一部分，修改 reward 时需要同时检查环境 reward 和训练 replay reward 的语义是否一致。

## 训练语义

当前 offset 训练流程如下：

1. 环境 `reset()` 返回 100 ms PSD waterfall observation。
2. 训练脚本固定 hoprate。
3. SAC actor 按顺序生成 10 个 offset；第 `i` 个 offset 的输入包含当前 observation、固定 hoprate 和 block index `i`。`take_action_sequence` 已实现为单次 batch 前向（10 个 block 一次性输入），推理效率较高。
4. 环境一次性执行这 10 个 offset，对应 10 个 100 ms block。
5. 环境返回下一个 observation，以及 `info["ber_blocks"]`。
6. 训练脚本把 10 个 block 拆成 10 条 replay transitions。

Replay 将这 10 个 block 建模为顺序状态：block 0～8 的下一状态保持当前 PSD/hoprate 并递增 block index，block 9 才进入环境返回的下一 PSD/hoprate 并回到 block 0。这仍是工程近似，因为环境一次性执行完整 offset 向量，而不是在每个 block 后重新观测。

## SAC 实现要点

当前 [SAC.py](SAC.py) / [SAC_test.py](special_hopping_test/SAC_test.py) 的实现要点（近期已修复若干问题）：

- **actor 与 critic 各自独立维护 `extra_embedding`**，不再共享；actor 优化器直接更新自己的 embedding。
- **目标网络固定为 eval 模式**：`target_critic_1/2` 始终用 BatchNorm running stats，不随 batch 抖动。
- **Lazy target 延迟初始化**：首次 TD target 计算前先物化 online critics，再完整复制参数和 BatchNorm buffers；后续只做 soft update。
- **`soft_update` 同步 BatchNorm buffers**（running_mean/running_var/num_batches_tracked）到目标网络。
- **`calc_target` 在 `torch.no_grad()` 下、actor 切 eval** 计算 next-state 目标，避免建图/污染 BN stats。
- **actor loss 计算时 critic 切 eval**，事后恢复 train，让策略梯度基于稳定的 running stats。
- **`take_action_sequence` 批量前向**，10 个 block 一次推理。

## 通信环境设计取舍

- [fh_env.py](fh_env.py) 中 `PreGeneratedData.common_bits` 生成并复用一份 bits，用于速度和内存优化。
- `reset_mseq_each_step=True` 的固定模板训练行为保留。
- `use_pregen=True` 默认使用预生成池和预生成 observation 加速训练。
- comb 干扰使用两组固定 8 信道频点交替：phase 0 使用偶数信道组，phase 1 使用奇数信道组。

如需做更严谨的通信环境对照实验，建议后续单独比较：`use_pregen=True/False`、固定/连续 m-sequence、不同 bits 随机化策略、不同 reward 权重、是否启用 Rayleigh/反应式/扫频干扰组合等。

## 已知限制和后续建议

- **项目尚未完成**：训练效果验证、算法稳定性、多模块联调仍在推进中。
- 10-offset replay 拆分是近似建模，不是严格序列 MDP。
- 训练效果对 reward 权重、alpha 初值、target entropy、batch size 等参数敏感。
- MBPO 奖励模型只预测 reward、不预测下一帧 PSD，合成样本复用真实 transition 已编码的顺序 next state；`real_ratio` 过低时可能放大模型偏差。
- NBS 跳速搜索依赖 BER-vs-hoprate 的可辨识趋势；若同时启用多种强干扰或随机 offset 方差很大，可能需要增加步数、调大 `p` 或做多次重复评估。
- `SAC.py` 中定义了 `EASReplayBuffer`，但当前没有训练入口使用它；EAS-local 训练流程属于后续计划。
- 若后续要进一步规范工程结构，可以再做第二阶段重构：拆分 `env/`、`algos/`、`train/` 子包，抽取公共训练工具函数，补充依赖文件和自动化测试。
