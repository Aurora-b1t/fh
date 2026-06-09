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
- 将局部搜索后得到的更优动作再蒸馏回 actor。

## 2. 核心思想

新增算法文件：[SAC_eas_local.py](SAC_eas_local.py)
新增训练入口：[train_offsets_eas_local.py](train_offsets_eas_local.py)

该方法借鉴了 Evolutionary Action Selection 的思想，但只保留最适合本项目的部分：

- 不做连续动作 PSO；
- 不做 10-offset 联合搜索；
- 只在**单个离散 offset 决策点**做局部搜索；
- 用 critic 对候选动作评分；
- 用搜索后的执行动作作为 teacher，对 actor 加蒸馏损失。

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

### A. 局部邻域搜索

对于 actor 采样得到的 seed action：

1. 构造局部邻域候选集：`seed_action ± search_radius`
2. 使用 twin critics 对这些候选动作逐个评分
3. 默认采用 `min(Q1, Q2)` 作为保守评分
4. 选择局部评分最高的动作作为最终执行动作

因此，执行动作不再总是 actor 原始采样值，而是：

- **actor proposal + local critic refinement**

### B. 蒸馏回 actor

训练时，除了保留原始 SAC actor loss，还加入一个蒸馏项：

- teacher 采用 replay 中记录的最终执行动作
- actor logits 对 teacher action 做 cross-entropy

总 actor loss 为：

```text
actor_total_loss = sac_actor_loss + distill_coef * distill_loss
```

其中：

- `sac_actor_loss`：原始离散 SAC actor objective
- `distill_loss`：对局部搜索执行动作的离散监督
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
- 不需要修改 replay schema
- 能保持 baseline 与改进版的可比性

## 5. replay 中记录什么动作

改进版中，replay 记录的是：

- **局部搜索后的最终执行动作**

而不是 actor 原始采样得到的 seed action。

这样做的理由：

1. replay 中动作必须与环境真实执行动作一致；
2. 蒸馏时可以直接把 replay action 当作 teacher；
3. 不需要给 replay 额外增加字段，保持数据结构与 baseline 一致。

## 6. 新增配置项

在 [settings.py](settings.py) 中新增了 `EAS_LOCAL_CONFIG`，用于控制改进版：

- `search_radius`：局部邻域半径
- `distill_coef`：蒸馏损失权重
- `search_eval`：候选动作评分方式，当前默认 `min_q`
- `teacher_from_replay`：是否用 replay 动作作为蒸馏 teacher
- `log_search_stats`：是否记录局部搜索统计

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
  - 局部搜索是否真的改写了 seed action
- `search_avg_gain`
  - 局部搜索相对 seed action 的平均 Q 提升
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
- 不在 update 阶段重新搜索 teacher；
- teacher 直接来自 replay 中的执行动作。

如果后续需要进一步增强，可以再研究：

- 更大的局部半径
- top-k actor proposal + local reranking
- update 阶段重新搜索 teacher
- sequence-level search / planner
- soft target distillation 而不是 hard CE

## 10. 文件对应关系

- 原始基线算法：[SAC.py](SAC.py)
- 原始训练脚本：[train_offsets_v1.py](train_offsets_v1.py)
- 改进算法实现：[SAC_eas_local.py](SAC_eas_local.py)
- 改进训练脚本：[train_offsets_eas_local.py](train_offsets_eas_local.py)
- 配置：[settings.py](settings.py)

该组织方式的目的，就是让 baseline 和改进版能够**并行存在、分别训练、直接对比**。
