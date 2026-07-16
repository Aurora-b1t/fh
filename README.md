# 强化学习跳频抗干扰系统

本项目是一个面向跳频扩频（FHSS, Frequency Hopping Spread Spectrum）抗干扰研究的 Python 实验环境。项目核心包含：

- **FHSS/QPSK 通信仿真环境**：QPSK 收发链路、跳频信道、PSD waterfall 观测、Rayleigh 衰落、扫频/梳状/反应式干扰机。
- **baseline 离散 SAC 训练**：agent 在每个环境 step 内自回归选择 10 个跳频 offset。
- **Noisy Binary Search 跳速阈值搜索**：用 MWU-based noisy binary search 寻找反应式干扰机的跳速跟踪/失效边界。
- **derivative NBS 变体**：用 BER-hoprate 导数指标代替单纯 BER 升降做方向决策。
- **hoprate sweep 评估**：确定性网格遍历所有候选 hoprate，作为 NBS 搜索的对照基线。

项目根目录采用平铺结构，入口清晰、文件职责清晰、运行方式清晰。每个训练/搜索脚本自带默认输出目录，互不覆盖。

## 项目文件说明

| 文件 | 作用 |
| --- | --- |
| [settings.py](settings.py) | 统一配置环境参数、干扰机参数、SAC/EAS/NBS 超参数、训练循环和奖励系数。输出目录由各训练脚本自行配置。 |
| [fh_env.py](fh_env.py) | FHSS/QPSK Gymnasium 环境，包含预生成加速、跳频序列、QPSK 收发处理、干扰/衰落叠加、BER 和 reward 计算。 |
| [jammers.py](jammers.py) | 干扰机实现：快速带限噪声源、基于能量检测的反应式干扰机、扫频/梳状宽带干扰机。 |
| [SAC.py](SAC.py) | baseline 离散动作 SAC 实现，包含 replay buffer、CNN policy/Q 网络、温度系数自适应和软更新。 |
| [train_offsets.py](train_offsets.py) | baseline 训练入口，负责构建环境、SAC agent、replay buffer、训练循环、日志和曲线输出。 |
| [train_speed.py](train_speed.py) | Noisy Binary Search 跳速阈值搜索入口，用随机 offset 和反应式干扰机评估 BER-vs-hoprate 边界。 |
| [train_speed_derivative.py](train_speed_derivative.py) | derivative-based NBS 变体入口，用 BER-hoprate 导数指标做方向决策。 |
| [train_speed_sweep.py](train_speed_sweep.py) | hoprate 网格扫描评估入口，确定性遍历所有候选 hoprate 并记录 BER/reward 诊断。 |
| [noisy_binary_search.py](noisy_binary_search.py) | MWU-based noisy binary search 算法实现，参考 Dereniowski et al. STACS 2025。 |
| [noisy_binary_search_derivative.py](noisy_binary_search_derivative.py) | derivative-based NBS 算法实现，用导数指标代替 BER 升降做方向决策。 |

历史文件说明：

- 旧训练脚本和启动脚本（如 `run.py`、`speedDQN.py`、`train_offsets.py` 早期版本、`Jamming Strategy.md` 等）已不作为当前根目录维护入口。
- 如需查看历史版本，可通过 git 历史记录找回。

## 运行环境

用户当前环境为：

```bash
D:\Anaconda\envs\rl_fhss\python.exe
```

代码中主要使用：

- `numpy`
- `matplotlib`
- `scipy`
- `gymnasium`
- `torch`
- `commpy`

建议在项目根目录运行所有命令。

## Baseline：离散 SAC offset 训练

查看参数：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets.py --help
```

短训练冒烟测试：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets.py --steps_per_episode 3 --batch_size 20 --offline_replay_path outputs/offline_replay/replay_50000_random_hoprate.npz --output_dir outputs/smoke
```

默认训练（默认输出目录 `outputs/offsets`）：

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

运行示例（默认输出目录 `outputs/speed_derivative`）：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_speed_derivative.py --steps 60 --output_dir outputs/speed_test_derivative
D:\Anaconda\envs\rl_fhss\python.exe train_speed_derivative.py --derivative_threshold -0.005 --steps 100
```

### hoprate 网格扫描评估

[train_speed_sweep.py](train_speed_sweep.py) 是 [train_speed.py](train_speed.py) 的确定性网格对照：不使用 NBS 选择下一个 hoprate，而是按升序遍历每个候选 hoprate，记录 BER/reward 诊断。每个环境 step 内部执行 10 个 block，与训练脚本的环境 step 语义一致。

运行示例（默认输出目录 `outputs/speed_sweep`）：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_speed_sweep.py --output_dir outputs/speed_sweep
```

冒烟测试：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_speed_sweep.py --hoprate_max 20 --steps_per_hoprate 1
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

- [Dereniowski 等 - 2025 - Noisy (Binary) Searching Simple, Fast and Correct.pdf](<Dereniowski 等 - 2025 - Noisy (Binary) Searching Simple, Fast and Correct.pdf>)
- [Energy_detection_of_unknown_deterministic_signals.pdf](Energy_detection_of_unknown_deterministic_signals.pdf)

## 输出目录

输出目录不再集中在 [settings.py](settings.py)，而是由各训练/搜索脚本自带默认 `--output_dir`，均可通过命令行覆盖：

| 脚本 | 默认 `--output_dir` |
| --- | --- |
| [train_offsets.py](train_offsets.py) | `outputs/offsets` |
| [train_speed.py](train_speed.py) | `outputs/speed` |
| [train_speed_derivative.py](train_speed_derivative.py) | `outputs/speed_derivative` |
| [train_speed_sweep.py](train_speed_sweep.py) | `outputs/speed_sweep` |

每个脚本还支持 `--log_file`（默认 `training_log.txt`，位于对应 `--output_dir` 内）。常见输出文件包括：

- `training_log.txt`：训练或搜索日志。
- `reward.png`：平均 step reward 曲线（offset 训练）。
- `ber.png`：平均 step BER 曲线。
- `loss.png`：actor/critic loss 曲线（offset 训练）。
- `hoprate.png`、`ber_vs_hoprate.png`、`nbs_weights.png`：NBS 搜索诊断图。
- `hoprate_sweep.csv`、`hoprate_sweep.npz`：sweep 评估数据。

`outputs/` 已加入 [.gitignore](.gitignore)，训练产物默认不进入版本控制。历史输出目录保留用于对比。

## 配置说明

主要配置集中在 [settings.py](settings.py)。

### 设备

- `CPU_ONLY`：是否强制使用 CPU。

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

### `EAS_LOCAL_CONFIG`

EAS-local 专用配置（保留供后续 EAS-local 训练入口使用）：

- `search_radius`：局部候选半径。
- `distill_coef`：teacher distillation loss 权重。
- `search_eval`：候选评分方式，当前支持 `min_q`。
- `teacher_from_replay`：是否启用 EAS replay teacher 蒸馏。
- `log_search_stats`：是否记录局部搜索统计。
- `eas_replay_capacity`、`eas_batch_size`：EAS replay 容量和 batch size。
- `filter_teacher_on_update`：训练阶段是否动态过滤已不优于 actor 的 teacher 样本。
- `teacher_compare_mode`：teacher 与当前 actor action 的比较方式。

### `BUFFER_CONFIG`

普通 replay buffer 配置：`capacity`、`batch_size`。

### `NBS_CONFIG`

Noisy Binary Search 配置：

- `p`：假设的噪声概率，需满足 `0 <= p < 0.5`。
- `delta`：收敛阈值；当最大权重 `>= 1 - delta` 时认为收敛。
- `hoprate_step`：候选 hoprate 离散步长，默认 10 Hz，与环境 `_apply_hoprate()` 的量化一致。
- `seed`：NBS 随机化查询的随机种子。

### `TRAIN_CONFIG`

训练循环配置：

- `steps_per_episode`：单次训练运行的环境 step 数。
- Offline replay is loaded before the first gradient update; configure its file with `OFFLINE_REPLAY_CONFIG` or `--offline_replay_path`.
- `update_iters_per_step`：每个环境 step 后的梯度更新次数。
- `fixed_hoprate`：offset 训练时使用的固定 hoprate。

### `REWARD_CONFIG`

训练脚本中逐 block reward 的计算参数：

```text
reward = base_reward - ber_penalty * BER - hoprate_penalty * hoprate
```

注意：环境 [fh_env.py](fh_env.py) 中的 step reward 当前为 `0.5 - mean_ber - hoprate * 0.0001`；训练脚本使用 `REWARD_CONFIG` 为每个 block 生成 replay reward。两者都是当前实验设计的一部分，修改 reward 时需要同时检查环境 reward 和训练 replay reward 的语义是否一致。

## 训练语义

当前 offset 训练流程如下：

1. 环境 `reset()` 返回 100 ms PSD waterfall observation。
2. 训练脚本固定 hoprate。
3. SAC actor 按顺序生成 10 个 offset；第 `i` 个 offset 的输入包含当前 observation、固定 hoprate 和 block index `i`。
4. 环境一次性执行这 10 个 offset，对应 10 个 100 ms block。
5. 环境返回下一个 observation，以及 `info["ber_blocks"]`。
6. 训练脚本把 10 个 block 拆成 10 条 replay transitions。

这是一种工程近似：每个子动作共享同一个环境前态和最终后态，通过 block index 区分 offset 的顺序。它便于复用当前离散 SAC 结构，但不是严格的 sequence-level MDP 建模。如果后续要更严格处理 10 个 offset 的时序决策，可以考虑 LSTM、Transformer encoder、sequence-level SAC/PPO 或显式 sequence policy。

## 通信环境设计取舍

- [fh_env.py](fh_env.py) 中 `PreGeneratedData.common_bits` 生成并复用一份 bits，用于速度和内存优化。
- `reset_mseq_each_step=True` 的固定模板训练行为保留。
- `use_pregen=True` 默认使用预生成池和预生成 observation 加速训练。
- comb 干扰使用两组固定 8 信道频点交替：phase 0 使用偶数信道组，phase 1 使用奇数信道组。

如需做更严谨的通信环境对照实验，建议后续单独比较：`use_pregen=True/False`、固定/连续 m-sequence、不同 bits 随机化策略、不同 reward 权重、是否启用 Rayleigh/反应式/扫频干扰组合等。

## 本次整理内容

1. **输出目录配置下沉**：从 [settings.py](settings.py) 移除 `OUTPUT_DIR` / `LOG_FILE`，改由各训练/搜索脚本在 `--output_dir` / `--log_file` 参数中自带默认值，避免不同实验线共用同一默认目录互相覆盖。
2. **文件命名规范化**：去除版本后缀，重命名：
   - `fh_env_opt_newest.py` → [fh_env.py](fh_env.py)
   - `train_offsets_v1.py` → [train_offsets.py](train_offsets.py)
   - `train_speed_new.py` → [train_speed_derivative.py](train_speed_derivative.py)（去 `new` 后缀，因与 `train_speed.py` 冲突，按其使用的 derivative-based NBS 算法命名）
3. **README 重写**：文件说明表与命令示例同步新文件名；删除对仓库中不存在的 `SAC_eas_local.py`、`train_offsets_eas_local.py`、`Evolutionary Action Selection Local Search.md` 的引用；输出目录一节改为按脚本列出默认目录。

## 已知限制和后续建议

- 10-offset replay 拆分是近似建模，不是严格序列 MDP。
- actor 当前共享 `critic_1.extra_embedding`，且 actor optimizer 不直接更新共享 embedding；若后续训练仍不稳定，可以考虑 actor/critic 使用独立 extra encoder。
- 训练效果对 reward 权重、alpha 初值、target entropy、batch size 等参数敏感。
- NBS 跳速搜索依赖 BER-vs-hoprate 的可辨识趋势；若同时启用多种强干扰或随机 offset 方差很大，可能需要增加步数、调大 `p` 或做多次重复评估。
- 若后续要进一步规范工程结构，可以再做第二阶段重构：拆分 `env/`、`algos/`、`train/` 子包，抽取公共训练工具函数，并补充依赖文件和自动化测试。
