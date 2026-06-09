# Evolutionary Action Selection Local Search

本文档说明新增的 **EAS 风格局部邻域搜索版离散 SAC**，以及它相对原始 [SAC.py](SAC.py) 的具体改动。

## 1. 背景与目标

当前项目中的原始离散 SAC 训练流程为：

1. 每个环境 step 内顺序生成 10 个 offset；
2. 每个 offset 决策输入为：
   - 当前 observation（PSD waterfall）
   - 固定 hoprate
   - 当前 10 维 `action_arr` 历史
3. actor 直接给出离散动作分布，并从中采样一个 offset；
4. 最终这 10 个 offset 一起送入环境执行。

原始离散 SAC 的优点是结构清晰、实现简单；但它每个单步 offset 的选择完全依赖 actor 当前分布，没有额外的基于 critic 的动作精修过程。

新增的改进版目标是：

- 保留原始离散 SAC 作为 baseline；
- 不改变“10 个 offset = 10 个顺序决策”的建模语义；
- 在每个单步 offset 决策时增加 **局部邻域搜索**；
- 不让局部搜索结果直接替换环境执行动作；
- 将局部搜索得到的更优动作作为 teacher 样本单独存储，并在训练时再利用。

## 2. 核心思想

新增算法文件：[SAC_eas_local.py](SAC_eas_local.py)
新增训练入口：[train_offsets_eas_local.py](train_offsets_eas_local.py)

该方法借鉴了 Evolutionary Action Selection 的思想，但只保留最适合本项目的部分：

- 不做连续动作 PSO；
- 不做 10-offset 联合搜索；
- 只在**单个离散 offset 决策点**做局部搜索；
- 用 critic 对候选动作评分；
- 环境仍执行 actor 原始采样动作；
- 将搜索得到的更优动作写入单独的 EAS replay，训练时再用于蒸馏 actor。

## 3. 与原始离散 SAC 的区别

## 3.1 原始离散 SAC

原始 [SAC.py](SAC.py) 中，`take_action(...)` 的行为是：

- actor 输出离散动作概率；
- 从 categorical 分布直接采样动作；
- 该动作直接作为执行动作。

训练时：

- critic 使用 standard twin-Q update；
- actor 使用 standard entropy-regularized discrete SAC objective；
- 没有额外的动作精修和蒸馏项。

## 3.2 新增 EAS 局部搜索版

在 [SAC_eas_local.py](SAC_eas_local.py) 中，对每个 offset 决策增加两步：

### A. 局部邻域搜索（仅生成 teacher 候选）

对于 actor 采样得到的 seed action：

1. 构造局部邻域候选集：`seed_action ± search_radius`
2. 使用 twin critics 对这些候选动作逐个评分
3. 默认采用 `min(Q1, Q2)` 作为保守评分
4. 选择局部评分最高的动作作为 `best_action`

但与旧版本不同的是：

- **环境执行的仍然是 actor 原始采样得到的 `seed_action`**
- `best_action` **不会直接用于环境执行**
- `best_action` 仅作为后续训练的 teacher 候选样本

因此这里的语义是：

- **actor proposal for execution + local critic teacher proposal for training**

### B. 单独 EAS replay + 训练阶段动态过滤

训练时，除了保留原始 SAC actor loss，还额外引入一个来自 EAS replay 的蒸馏项。

EAS replay 中记录的是：

- 当前状态 `(state_img, hoprate, action_arr)`
- 局部搜索得到的 `best_action`
- 可选统计信息（如 `search_gain`）

注意：

- 写入 EAS replay 时**不做过滤**；
- 真正是否使用该 teacher 样本，是在 **update 阶段** 动态决定的。

具体地说：

1. 从 EAS replay 采样 teacher 样本；
2. 对每条样本，先用**当前 actor**在该状态下给出当前动作（实现上可固定取 `argmax` 以避免额外采样噪声）；
3. 用当前 critic 比较：
   - replay 中的 `best_action`
   - 当前 actor 动作
4. 只有当 `best_action` 在当前 critic 视角下仍然更优时，该样本才保留蒸馏梯度；否则本次 mask 掉。

总 actor loss 为：

```text
actor_total_loss = sac_actor_loss + distill_coef * masked_distill_loss
```

其中：

- `sac_actor_loss`：原始离散 SAC actor objective
- `masked_distill_loss`：仅对“当前仍优于 actor 动作”的 teacher 样本计算的监督损失
- `distill_coef`：蒸馏权重

## 4. 为什么只做单步局部搜索

用户已明确当前项目中：

- 10 个 offset 就是 10 个顺序步骤
- 不把 10 个 offset 作为联合动作处理

因此本算法只在**每一个 offset 决策点**做局部搜索，而不是：

- 不做 10 维组合搜索
- 不做 beam search
- 不做 sequence-level planner

这样做的好处是：

- 与现有 trainer 完全兼容
- 不需要修改环境结构
- 能保持 baseline 与改进版的可比性

## 5. 两类 replay 各自记录什么

### 普通 replay

普通 replay 中记录的是：

- **环境真实执行动作 `seed_action` 对应的 transition**

它继续服务于：

- critic TD 学习
- 标准 SAC actor 学习

### EAS replay

EAS replay 中记录的是：

- **局部搜索得到的 `best_action` teacher 样本**

它只服务于：

- actor 的额外蒸馏监督

因此，相比 baseline：

- 普通 replay 语义不变；
- 但新增了一个单独的 teacher replay buffer。

## 6. 新增配置项

在 [settings.py](settings.py) 中新增或扩展 `EAS_LOCAL_CONFIG`，用于控制改进版：

- `search_radius`：局部邻域半径
- `distill_coef`：蒸馏损失权重
- `search_eval`：候选动作评分方式，当前默认 `min_q`
- `teacher_from_replay`：是否启用来自 EAS replay 的 teacher 蒸馏
- `log_search_stats`：是否记录局部搜索统计
- `eas_replay_capacity`：EAS replay 容量
- `eas_batch_size`：EAS replay 训练 batch size
- `filter_teacher_on_update`：是否在训练阶段按当前 critic 判断 teacher 是否仍占优
- `teacher_compare_mode`：teacher 与当前 actor 动作的比较方式

推荐初始值：

- `search_radius = 1`
- `distill_coef = 0.1`

## 7. 训练入口与对比方式

### Baseline

原始离散 SAC：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets_v1.py --output_dir outputs/baseline
```

### 改进版

EAS 局部邻域搜索版：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets_eas_local.py --output_dir outputs/eas_local
```

## 8. 改进版新增的可观测指标

相比 baseline，改进版建议重点观察：

- `distill_loss`
- `search_change_rate`
  - 局部搜索是否真的改写了 teacher 候选
- `search_avg_gain`
  - 局部搜索相对 seed action 的平均 Q 提升
- `eas_valid_ratio`
  - 训练阶段从 EAS replay 采样后，仍被当前 critic 认为优于 actor 动作、因此保留梯度的样本比例
- 与 baseline 对比的：
  - mean step reward
  - mean BER
  - critic/actor loss
  - wall-clock time

## 9. 适用边界

该方法是对原始离散 SAC 的一个轻量增强，而不是完全重写算法。

当前版本有以下明确边界：

- 不修改环境结构；
- 不改变 10-step 顺序决策语义；
- 不做联合 10-offset 搜索；
- 执行动作始终来自 actor 采样；
- 局部搜索结果只进入单独 EAS replay；
- teacher 是否有效在 update 阶段动态过滤。

如果后续需要进一步增强，可以再研究：

- 更大的局部半径
- top-k actor proposal + local reranking
- 更丰富的 teacher 过滤标准
- sequence-level search / planner
- soft target distillation 而不是 hard CE

## 10. 文件对应关系

- 原始基线算法：[SAC.py](SAC.py)
- 原始训练脚本：[train_offsets_v1.py](train_offsets_v1.py)
- 改进算法实现：[SAC_eas_local.py](SAC_eas_local.py)
- 改进训练脚本：[train_offsets_eas_local.py](train_offsets_eas_local.py)
- 配置：[settings.py](settings.py)

该组织方式的目的，就是让 baseline 和改进版能够**并行存在、分别训练、直接对比**。
