# 强化学习跳频抗干扰系统

本项目是一个基于强化学习的跳频扩频（FHSS）抗干扰训练实验环境。当前主线使用 QPSK 收发链路、跳频信道、扫频/梳状/响应式干扰机、Rayleigh 衰落以及离散动作 SAC 算法，训练 agent 为每个环境步选择 10 个跳频 offset。

## 当前主线文件

| 文件 | 作用 |
| --- | --- |
| [train_offsets_v1.py](train_offsets_v1.py) | 当前推荐训练入口，负责构建环境、SAC agent、replay buffer、训练循环、日志和曲线输出。 |
| [settings.py](settings.py) | 统一配置环境、干扰机、SAC、replay buffer、训练步数和奖励参数。 |
| [fh_env_opt_newest.py](fh_env_opt_newest.py) | FHSS/QPSK 通信环境，包含预生成加速、跳频序列、收发处理、BER 和 reward 计算。 |
| [SAC.py](SAC.py) | 离散动作 SAC 实现，状态由 PSD 图像、固定 hoprate 和 10 维 action history 组成。 |
| [jammers.py](jammers.py) | 响应式干扰机和扫频/梳状宽带干扰机实现。 |

历史文件说明：

- [speedDQN.py](speedDQN.py)：跳速调节 DQN 方案，已废弃，本项目当前整理中不使用。
- [train_offsets.py](train_offsets.py)：旧训练脚本，引用了当前不存在的 `fh_env`，不作为当前入口。
- [run.py](run.py)：旧启动脚本，仍引用旧包路径 `fh.newray` / `fh.speedDQN`，不作为当前入口。

## 运行环境

用户当前环境为：

```bash
D:\Anaconda\envs\rl_fhss\python.exe
```

依赖已安装。代码中主要使用：

- `numpy`
- `matplotlib`
- `scipy`
- `gymnasium`
- `torch`
- `commpy`

## 快速运行

在项目根目录运行。

查看参数：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets_v1.py --help
```

短训练冒烟测试：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets_v1.py --steps_per_episode 3 --min_buffer_before_train 20 --output_dir outputs/smoke
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
D:\Anaconda\envs\rl_fhss\python.exe train_offsets_v1.py --output_dir outputs/experiment_001
```

## 输出文件

训练输出不再写入项目根目录，默认写入：

```text
outputs/latest/
```

每次训练会生成：

- `training_log.txt`：训练日志
- `reward.png`：平均 step reward 曲线
- `ber.png`：平均 step BER 曲线
- `loss.png`：actor/critic loss 曲线

根目录历史训练输出已整理到：

```text
outputs/archive_root_outputs/
```

`outputs/` 已加入 [.gitignore](.gitignore)，训练产物默认不进入版本控制。

## 配置说明

主要配置集中在 [settings.py](settings.py)：

- `OUTPUT_DIR` / `LOG_FILE`：默认输出目录和日志文件名。
- `ENV_CONFIG`：频段、采样率、码元率、干扰/衰落开关、预生成设置。
- `JAMMER_CONFIG`：扫频、梳状、响应式干扰参数。
- `SAC_CONFIG`：actor/critic/alpha 学习率、软更新系数、折扣因子、目标熵比例。
- `BUFFER_CONFIG`：replay buffer 容量和 batch size。
- `TRAIN_CONFIG`：训练步数、warmup replay entries、每步更新次数、固定 hoprate。
- `REWARD_CONFIG`：训练分块奖励参数，已与环境 step reward 尺度保持一致。

注意：`min_buffer_before_train` 表示 replay entries 数量，不是环境 step 数。当前一个环境 step 会拆成 10 条 replay transition，因此默认 `1200` 大约对应 120 个环境 step 后开始训练。

## 训练语义

当前训练流程如下：

1. 环境 reset 返回一个 100 ms PSD waterfall observation。
2. SAC agent 基于当前 observation、固定 hoprate 和 10 维 action history，自回归采样 10 个 offset。
3. 环境一次性执行这 10 个 offset，对应 10 个 block。
4. 环境返回下一个 observation，以及 `info["ber_blocks"]`。
5. 训练脚本把这 10 个 block 拆成 10 条 replay transition。

这是一种工程近似：每个子动作共享同一个环境前态和最终后态，通过 action history 区分 10 个 offset 的顺序。它便于复用当前离散 SAC 结构，但并不是严格的 sequence-level MDP 建模。如果后续要更严格处理 10 个 offset 的时序决策，可以考虑 LSTM、Transformer encoder、sequence-level SAC/PPO 等结构。

## 通信环境设计取舍

本次整理只检查并记录通信环境，不改变下列实验设定：

- [fh_env_opt_newest.py](fh_env_opt_newest.py) 中 `PreGeneratedData.common_bits` 只生成并复用一份 bits。这是为了优化速度和内存占用。
- `reset_mseq_each_step=True` 的固定模板训练行为保留，这是当前实验设计的一部分。
- `use_pregen=True` 使用预生成池和预生成 observation 来加速训练，这是当前默认加速路径。

如需做更严谨的通信环境对照实验，建议后续单独比较 `use_pregen=True/False`、固定/连续 m-sequence、不同 bits 随机化策略等设置。

## 本次整理和修正

本次改动包含：

1. 训练输出整理
   - 新增 `--output_dir` 参数。
   - 日志和三张训练曲线统一写入 `outputs/...`。
   - 根目录已有训练产物移动到 `outputs/archive_root_outputs/`。
   - [.gitignore](.gitignore) 增加 `outputs/`。

2. SAC 关键稳定性修复
   - 修正 alpha/temperature 自适应更新方向。
   - `take_action()` 使用 `torch.no_grad()`，并在采样时临时切换 `actor.eval()`，避免 BatchNorm 被单样本在线推理污染统计量。

3. 奖励语义统一
   - 训练脚本的分块 reward 与环境 step reward 保持同一尺度：`base_reward - ber_penalty * BER - hoprate_penalty * hoprate`。
   - 当前默认对应环境公式：`0.5 - BER - hoprate * 0.0001`。

4. 注释修正
   - [jammers.py](jammers.py) 中 comb 干扰注释已修正为真实实现：phase 0 使用偶数通道组，phase 1 使用奇数通道组。

5. 文档补充
   - 增加本 README，说明当前入口、运行方式、输出目录、训练语义、历史文件和已知设计取舍。

## 已知限制和后续建议

- 10-offset replay 拆分是近似建模，不是严格序列 MDP。
- actor 当前共享 `critic_1.extra_embedding`，且 actor optimizer 不直接更新共享 embedding；若后续训练仍不稳定，可以考虑 actor/critic 使用独立 extra encoder。
- 旧入口文件仍保留，但不建议继续使用。
- 若训练效果不理想，建议下一步优先对比：奖励权重、alpha 初值/目标熵、是否取消 extra embedding 共享、是否改成真正序列策略模型。
