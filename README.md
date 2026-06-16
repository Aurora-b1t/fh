# 强化学习跳频抗干扰系统

本项目是一个面向跳频扩频（FHSS, Frequency Hopping Spread Spectrum）抗干扰研究的 Python 实验环境。项目核心包含：

- **FHSS/QPSK 通信仿真环境**：QPSK 收发链路、跳频信道、PSD waterfall 观测、Rayleigh 衰落、扫频/梳状/反应式干扰机。
- **baseline 离散 SAC 训练**：agent 在每个环境 step 内自回归选择 10 个跳频 offset。
- **EAS-local 改进训练**：在每个单步 offset 决策点引入局部邻域搜索，并用 teacher action 蒸馏改进 actor。
- **Noisy Binary Search 跳速阈值搜索**：用 MWU-based noisy binary search 寻找反应式干扰机的跳速跟踪/失效边界。

当前整理方案保持项目根目录平铺结构不变，重点保证：入口清晰、文件职责清晰、运行方式清晰、历史文件与实验限制说明清晰。

## 项目文件说明

| 文件 | 作用 |
| --- | --- |
| [settings.py](settings.py) | 统一配置输出目录、环境参数、干扰机参数、SAC/EAS/NBS 超参数、训练循环和奖励系数。 |
| [fh_env_opt_newest.py](fh_env_opt_newest.py) | FHSS/QPSK Gymnasium 环境，包含预生成加速、跳频序列、QPSK 收发处理、干扰/衰落叠加、BER 和 reward 计算。 |
| [jammers.py](jammers.py) | 干扰机实现：快速带限噪声源、基于能量检测的反应式干扰机、扫频/梳状宽带干扰机。 |
| [SAC.py](SAC.py) | baseline 离散动作 SAC 实现，包含 replay buffer、CNN policy/Q 网络、温度系数自适应和软更新。 |
| [SAC_eas_local.py](SAC_eas_local.py) | EAS-local 离散 SAC 变体：actor 采样动作，critic 做局部候选评分，训练阶段用 teacher action 蒸馏。 |
| [train_offsets_v1.py](train_offsets_v1.py) | baseline 训练入口，负责构建环境、SAC agent、replay buffer、训练循环、日志和曲线输出。 |
| [train_offsets_eas_local.py](train_offsets_eas_local.py) | EAS-local 训练入口，与 baseline 并行存在，可分别训练、直接对比。 |
| [train_speed.py](train_speed.py) | Noisy Binary Search 跳速阈值搜索入口，用随机 offset 和反应式干扰机评估 BER-vs-hoprate 边界。 |
| [noisy_binary_search.py](noisy_binary_search.py) | MWU-based noisy binary search 算法实现，参考 Dereniowski et al. STACS 2025。 |
| [Evolutionary Action Selection Local Search.md](Evolutionary%20Action%20Selection%20Local%20Search.md) | EAS-local 方法设计文档，说明与 baseline SAC 的区别、teacher replay、蒸馏指标和适用边界。 |

历史文件说明：

- 旧训练脚本和启动脚本（如 `run.py`、`speedDQN.py`、`train_offsets.py`、`Jamming Strategy.md` 等）已不作为当前根目录维护入口。
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
D:\Anaconda\envs\rl_fhss\python.exe train_offsets_v1.py --help
```

短训练冒烟测试：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets_v1.py --steps_per_episode 3 --min_buffer_before_train 20 --batch_size 20 --output_dir outputs/smoke
```

默认训练：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets_v1.py
```

强制 CPU：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets_v1.py --cpu_only
```

指定输出目录：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets_v1.py --output_dir outputs/baseline
```

## EAS-local：局部搜索增强版 SAC 训练

EAS-local 在 baseline 离散 SAC 的基础上增加局部邻域搜索：actor 先给出 seed action，critic 对 `seed_action ± search_radius` 的局部候选评分；局部最优候选不直接替换环境执行动作，而是写入单独的 EAS replay，在训练阶段作为 teacher action 蒸馏 actor。

查看参数：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets_eas_local.py --help
```

短训练冒烟测试：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets_eas_local.py --steps_per_episode 3 --min_buffer_before_train 20 --batch_size 20 --eas_batch_size 20 --output_dir outputs/eas_local_smoke
```

默认训练：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets_eas_local.py --output_dir outputs/eas_local
```

常用参数：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets_eas_local.py --search_radius 2 --distill_coef 0.5 --output_dir outputs/eas_local_r2
```

相比 baseline，EAS-local 额外输出：

- `search_change_rate.png`：局部搜索将 teacher action 改为非 seed action 的比例。
- `eas_valid_ratio.png`：训练阶段 teacher action 仍被当前 critic 判断优于 actor action 的比例。

详细设计见 [Evolutionary Action Selection Local Search.md](Evolutionary%20Action%20Selection%20Local%20Search.md)。

## 跳速阈值搜索（Noisy Binary Search）

[train_speed.py](train_speed.py) 使用 [noisy_binary_search.py](noisy_binary_search.py) 中的 MWU-based noisy binary search 算法，在反应式干扰机开启时搜索跳速阈值。每个环境 step 中：

1. NBS 给出一个待测试 hoprate。
2. 训练脚本随机生成 10 个 offset（不训练 SAC）。
3. 环境执行 10 个 block 并返回 BER。
4. NBS 根据当前 BER 与上一次 BER 的变化更新候选跳速权重分布。

运行示例：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_speed.py --steps 60 --output_dir outputs/speed_test
```

自定义 NBS 参数：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_speed.py --nbs_p 0.15 --nbs_delta 0.03 --nbs_step 10 --steps 100 --output_dir outputs/speed_custom
```

主要输出：

- `training_log.txt`：搜索日志。
- `hoprate.png`：实际测试 hoprate 与 NBS 估计轨迹。
- `ber.png`：每 step 平均 BER。
- `ber_vs_hoprate.png`：BER 与 hoprate 散点图。
- `nbs_weights.png`：最终候选 hoprate 权重分布。
- `nbs_distribution.npz`：候选集合、权重、测试 hoprate、BER 的 numpy 数据。

相关参考：

- [Dereniowski 等 - 2025 - Noisy (Binary) Searching Simple, Fast and Correct.pdf](<Dereniowski 等 - 2025 - Noisy (Binary) Searching Simple, Fast and Correct.pdf>)
- [Energy_detection_of_unknown_deterministic_signals.pdf](Energy_detection_of_unknown_deterministic_signals.pdf)

## 输出目录

训练和搜索输出默认写入 [settings.py](settings.py) 中的 `OUTPUT_DIR`：

```text
outputs/latest/0.1ms
```

也可以通过命令行 `--output_dir` 覆盖。常见输出文件包括：

- `training_log.txt`：训练或搜索日志。
- `reward.png`：平均 step reward 曲线（offset 训练）。
- `ber.png`：平均 step BER 曲线。
- `loss.png`：actor/critic/distill loss 曲线（训练入口）。
- `search_change_rate.png`：EAS-local 局部搜索改写比例。
- `eas_valid_ratio.png`：EAS-local teacher 样本有效比例。
- `hoprate.png`、`ber_vs_hoprate.png`、`nbs_weights.png`：NBS 搜索诊断图。

`outputs/` 已加入 [.gitignore](.gitignore)，训练产物默认不进入版本控制。历史输出目录保留用于对比，例如 `outputs/archive_root_outputs/`、`outputs/smoke/`、`outputs/eas_local_smoke/` 等。

## 配置说明

主要配置集中在 [settings.py](settings.py)。

### 输出与设备

- `OUTPUT_DIR`：默认输出目录。
- `LOG_FILE`：默认日志文件名。
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

baseline 与 EAS-local 共用的 SAC 超参数：

- `actor_lr`、`critic_lr`、`alpha_lr`
- `tau`
- `gamma`
- `target_entropy_ratio`

### `EAS_LOCAL_CONFIG`

EAS-local 专用配置：

- `search_radius`：局部候选半径。
- `distill_coef`：teacher distillation loss 权重。
- `search_eval`：候选评分方式，当前支持 `min_q`。
- `teacher_from_replay`：是否启用 EAS replay teacher 蒸馏。
- `log_search_stats`：是否记录局部搜索统计。
- `eas_replay_capacity`、`eas_batch_size`：EAS replay 容量和 batch size。
- `filter_teacher_on_update`：训练阶段是否动态过滤已不优于 actor 的 teacher 样本。
- `teacher_compare_mode`：teacher 与当前 actor action 的比较方式。

### `BUFFER_CONFIG`

普通 replay buffer 配置：

- `capacity`：普通 replay 容量。
- `batch_size`：SAC 更新 batch size。

### `NBS_CONFIG`

Noisy Binary Search 配置：

- `p`：假设的噪声概率，需满足 `0 <= p < 0.5`。
- `delta`：收敛阈值；当最大权重 `>= 1 - delta` 时认为收敛。
- `hoprate_step`：候选 hoprate 离散步长，默认 10 Hz，与环境 `_apply_hoprate()` 的量化一致。
- `seed`：NBS 随机化查询的随机种子。

### `TRAIN_CONFIG`

训练循环配置：

- `steps_per_episode`：单次训练运行的环境 step 数。
- `min_buffer_before_train`：开始梯度更新前的 replay entry 数量。注意一个环境 step 会拆成 10 条 replay transition。
- `update_iters_per_step`：每个环境 step 后的梯度更新次数。
- `fixed_hoprate`：offset 训练时使用的固定 hoprate。

### `REWARD_CONFIG`

训练脚本中逐 block reward 的计算参数：

```text
reward = base_reward - ber_penalty * BER - hoprate_penalty * hoprate
```

注意：环境 [fh_env_opt_newest.py](fh_env_opt_newest.py) 中的 step reward 当前为 `0.5 - mean_ber - hoprate * 0.0001`；训练脚本使用 `REWARD_CONFIG` 为每个 block 生成 replay reward。两者都是当前实验设计的一部分，修改 reward 时需要同时检查环境 reward 和训练 replay reward 的语义是否一致。

## 训练语义

当前 offset 训练流程如下：

1. 环境 `reset()` 返回 100 ms PSD waterfall observation。
2. 训练脚本固定 hoprate，从全零 action history 开始。
3. SAC actor 按顺序生成 10 个 offset；第 `i` 个 offset 的输入包含当前 observation、固定 hoprate 和已经生成的 action history。
4. 环境一次性执行这 10 个 offset，对应 10 个 100 ms block。
5. 环境返回下一个 observation，以及 `info["ber_blocks"]`。
6. 训练脚本把 10 个 block 拆成 10 条 replay transition。

这是一种工程近似：每个子动作共享同一个环境前态和最终后态，通过 10 维 action history 区分 offset 的顺序。它便于复用当前离散 SAC 结构，但不是严格的 sequence-level MDP 建模。如果后续要更严格处理 10 个 offset 的时序决策，可以考虑 LSTM、Transformer encoder、sequence-level SAC/PPO 或显式 sequence policy。

## 通信环境设计取舍

本次整理只清理文档和无行为影响的调试残留，不改变下列实验设定：

- [fh_env_opt_newest.py](fh_env_opt_newest.py) 中 `PreGeneratedData.common_bits` 生成并复用一份 bits，用于速度和内存优化。
- `reset_mseq_each_step=True` 的固定模板训练行为保留。
- `use_pregen=True` 默认使用预生成池和预生成 observation 加速训练。
- comb 干扰使用两组固定 8 信道频点交替：phase 0 使用偶数信道组，phase 1 使用奇数信道组。
- EAS-local 中局部搜索结果只进入 teacher replay，不直接替换环境执行动作。

如需做更严谨的通信环境对照实验，建议后续单独比较：`use_pregen=True/False`、固定/连续 m-sequence、不同 bits 随机化策略、不同 reward 权重、是否启用 Rayleigh/反应式/扫频干扰组合等。

## 本次整理内容

本次整理包含：

1. 代码清理
   - 清理 [fh_env_opt_newest.py](fh_env_opt_newest.py) 中残留的调试计时代码和已注释打印。
   - 删除本地 `.claude/worktrees/` 残留工作树目录。
   - 删除 `__pycache__/` 中过期 `.pyc` 缓存。
   - 为主要 `.py` 文件添加模块级 docstring。

2. 文档更新
   - README 补充 baseline、EAS-local、NBS 三条当前实验线。
   - README 补充 EAS-local 额外指标与 NBS 输出文件。
   - README 更新历史文件说明，不再把已删除旧入口描述为当前可运行入口。
   - README 补充相关论文 PDF 链接。

## 已知限制和后续建议

- 10-offset replay 拆分是近似建模，不是严格序列 MDP。
- actor 当前共享 `critic_1.extra_embedding`，且 actor optimizer 不直接更新共享 embedding；若后续训练仍不稳定，可以考虑 actor/critic 使用独立 extra encoder。
- 训练效果对 reward 权重、alpha 初值、target entropy、batch size、是否启用 EAS teacher 过滤等参数敏感。
- NBS 跳速搜索依赖 BER-vs-hoprate 的可辨识趋势；若同时启用多种强干扰或随机 offset 方差很大，可能需要增加步数、调大 `p` 或做多次重复评估。
- 若后续要进一步规范工程结构，可以再做第二阶段重构：拆分 `env/`、`algos/`、`train/` 子包，抽取公共训练工具函数，并补充依赖文件和自动化测试。
