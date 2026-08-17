# Goal-pose prior — v2 改动记录

分支 `feat/goal-prior-bottleneck-fix`,工作树 `/data2/JM/Code/ImageWAM-v2`
基线 commit `e81335b` = **被 LIBERO / LIBERO-Plus 评测的那个版本**(warmup 仍是 5000)

> 原仓库 `/data2/JM/Code/ImageWAM` 未被触碰 —— 评测的 32 个 worker 全程 spawn 新进程并
> import 源码,改原工作树会让新旧代码混进同一次对比。

---

## 为什么改:诊断

LIBERO-Plus 全量评测(n≈6000,`libero_10` 与 `libero_goal` 两个完整套件):

| 类别 | n | baseline | ours | Δ | 配对(赢/输) |
|---|---:|---:|---:|---:|---|
| Robot Initial States | 1152 | 51.0% | **62.7%** | **+11.6** | +215/−81 |
| Objects Layout | 737 | 68.0% | **72.0%** | **+4.1** | +72/−42 |
| Background Textures | 828 | 86.2% | **89.5%** | **+3.3** | +101/−74 |
| Light Conditions | 553 | 97.6% | 93.1% | −4.5 | +13/−38 |
| Sensor Noise | 828 | 96.0% | 90.1% | −5.9 | +17/−66 |
| Camera Viewpoints | 1099 | 76.3% | 65.9% | −10.5 | +79/−194 |
| Language Instructions | 804 | 88.1% | 74.3% | −13.8 | +23/−134 |
| **合计** | 6001 | 78.1% | 76.3% | **−1.8** | p=0.0014 |

剔除 Language 后是 **−0.6pp(p=0.33)**,统计上无差异。

**诊断:Action Expert 没有干净的退路。** syn 是它唯一的空间证据来源;任何污染
aggregator 输入的扰动都会污染它,而 baseline 可以退回去重读 784 个图像 token。
四个负项全是**近乎单向溃败**(不是互有胜负),这是依赖的签名,不是表征质量的问题。

**关键结构事实:ImageWAM 的 baseline 远比 MolmoAct2 的鲁棒**
(Sensor Noise 96.0% vs 44.7%,Camera 76.3% vs 32.3%)。MolmoAct2 上 V3 的 +8.4pp
很大程度是在修一个崩掉的 baseline;ImageWAM 没有那些弱点可修。

**赢的条件(按当前 n 加权,可精确复现 −1.8pp):**

| 情形 | 总体 Δ |
|---|---:|
| 现状 | −1.8pp |
| 中和 Camera + Sensor + Light | **+1.3pp** |
| 再中和 Language | **+3.2pp** |

图像侧那三项(n=2480,加权缺口 −18914)比 Language(−11095)更大,而且**它们的
text 通道是干净的**(扰动在图像侧),所以退路真实存在。

---

## 改了什么

### 主力 — 给 AE 造退路

**B2+ context 整段屏蔽 `context_blackout_prob: 0.10`**
以 10% 概率屏蔽全部 92 个 context token,**8 个 pose token 永远保留**。
屏蔽后 AE 退到 `[txt | pose(8) | action]` —— **恰好是 Stage1 先验的接口**。
逐样本采样,所以一个 batch 同时包含"被引导"和"退路"两种模式。

**B2 逐 token dropout `context_token_dropout: 0.15`**
打断对任何单个 context token 的依赖。

- 计算成本 **0**(只是 attention mask)
- 概率刻意压低:context 是承重的(MolmoAct2 上完全去掉会崩),
  而 Robot Initial States 的 +11.6pp 多半就走这条通道

### gate 而非 augment — 让先验活到 Stage2

**B4 `zero_init_value: true` + `syn_gate_bias_init: -5.0`**
`to_value` 零初始化 **且** 在 syn 列上加可学习的 logit bias。
只做零初始化是不够的 —— syn 列仍会从 text/action 那里抢走 softmax 质量,
只有加性 logit bias 才让这个通道在初始化时成为真正的 no-op。
训练路径和 cache 推理路径都接了(否则是训练/推理不一致)。

> **局限,必须记住**:门控只保证 step 0 时是关的。**训练是 in-distribution 的,
> syn 在那里永远有用,所以门会被一路开满。** B4 修的是交接,不是 OOD 损失。

**A1 `stage1_sample_video_timestep: true`**
`double_stream_modulation_txt(vec)` 用 video timestep 调制**整条 text stream**;
Stage1 原本恒 `t=0`,Stage2 遍历全程。纯分布错配,3 行。

**A2 `stage1_null_image_tokens: 32`**
Stage1 的图像流原本是**长度 0**(`new_zeros(B, 0, C)`),txt 除了自己什么都
attend 不到;Stage2 突然给它 784 个。加 32 个恒定可学习 token 稳住注意力形状。
**作为 ref 注入**,因为 `_build_mot_attention_mask_flux2` 里 txt 只 attend
`txt+ref`,放进 target 槽位这个改动会完全失效。

### 公平性与成本

**A4 `warmup_steps: 5000 → null`** → 走 5% 规则 = **1736**,与 baseline 逐值相同。
全程积分 LR 只差 0.09%,不构成重训理由,但消掉一个现成的质疑点。

**S1-1 Stage1 `batch_size: 128 → 64`**
Stage1 实测 **28.1h / 10k 步**(0.099 step/s,恒定,计算受限)。
loss 到 9k 还在降 34%,**不能砍步数**;砍 batch 保留全部 10k 次更新,
epoch 从 36.9 降到 18.4(数据集 277,760 样本),预计 **~17h,省 13h**。
LR 保持 2e-4 不降,抵消 batch 减半。

---

## 明确不做

| | 为什么 |
|---|---|
| **B1** 粗空间网格 | 代码在,**默认 `latent_layout: free`,本轮不启用**。网格把 latent 绑到局部窗口,但视角变化是**全局**几何变换,反而更脆 —— 而 Camera 是最大伤口 |
| **B3** L_inv | 针对 Light/Sensor,而 baseline 在那儿 96~98%,没有空间可赢,却要付一次额外前向 |
| **L_prior** anchor | 与 B4 功能重叠。开训 1 小时内看验收信号再定,省 10~15h。**不要用权重空间 anchor** —— 实测先验失效时权重漂移 <0.02%,"权重保住了、功能没保住" |
| **paraphrase 增强** | 会让 Language 的结果失去证据价值 —— baseline 没有,赢了也只说明增强有用 |

---

## 不变量

**所有开关默认还原被评测的行为**,`test_defaults_match_evaluated_revision` 专门守这一点。
不翻开关时,这个分支复现对照组。

---

## 测试

`tests/` 39 passed。新增覆盖:

- 默认值等价于被评测版本
- context dropout 豁免 pose token;**默认跟随模块 train/eval 模式**
  (曾经默认 `True`,会在推理时静默丢 token —— 已修 + 回归测试)
- blackout 是整段而非部分;与逐 token dropout 可叠加
- 门控 mask:被 mask 的位置 softmax 概率恰好为 0;syn 列被压低而 txt/action 不受影响
- `_format_attention_mask` 保留 float、bool 行为不变
- 网格几何(端到端:扰动左上角图像 token,右下角 latent 位移 < 1e-5)
- null token 以 ref 时间值 10.0 注入,4×8 位置网格

---

## 待做

- **A3** goal token 改走 syn 投影(方案待确认后实现)
- **S2-1** 验证 bridge 能加载 A3 的投影
- **D-2** 端到端 smoke:Stage1/Stage2 各跑一个训练步

## 开训前的探针(用现有权重,零训练)

| | 测什么 | 决定什么 |
|---|---|---|
| **P2** | in-dist 屏蔽 context 后成功率(现 97.4%) | **blackout 方案的硬前提**。掉到 90% → 退路可用;掉到 20% → 退路是空的 |
| **P1** | Language 子集屏蔽 context 后成功率 | 因果检验:若**上升**,诊断被证实 |
| **P3** | paraphrase 下 pose 漂移 vs context 漂移 | 若 pose 也被污染,要改 pose 头而非加 blackout |

---

## 验收信号

- **A 组 + B4**:Stage2 开局 loss 应远低于 **0.80**,且领先维持 **> 1000 步**
  (旧版是开局 0.80、step 100 被 baseline 反超)。不达标 → 补预计算版 L_prior
- **B2+**:Camera Viewpoints / Sensor Noise 回到不劣于 baseline
- **B2+**:Language 的配对不再单向(旧版 libero_10 上是 **+1 / −52**)

---

## 修订 1 — 门控只作用于 context,pose 保持常开

**发现的不一致**:B4 原本把门控加在全部 100 列上,于是 Stage2 在 step 0 时
AE 看到的是 `[txt | (syn 全关) | action]`。但 Stage1 的先验是
`π(a | txt, goal_tokens)` —— **位姿条件的**。把 syn 整个关掉,等于用一个
从未见过的输入(没有位姿)去跑一个位姿条件策略,**正是我们测到过的
"权重保住了、功能没保住"**。

而且它和 B2+ 自相矛盾:blackout 的兜底状态保留 pose,B4 的冷启动却把 pose 也关了。

**改法**:`gate_pose_tokens: false`。门控只覆盖 92 个 context 列。

| 通道 | 门控 | step 0 |
|---|---|---|
| pose(8) | 无 | 常开,携带推断位姿 |
| context(92) | logit bias −5 | 关闭,需自己挣到位置 |

三件事一次对齐:step-0 拓扑 = Stage1 先验接口;与 B2+ 兜底状态一致;
字面落实 "gate 而非 augment"(pose 是 steering 信号本身,context 才是 augment)。

**连带修正**:`zero_init_value` 必须设为 `false`。`to_value` 是所有 syn token
**共用**的一个 Linear,没有逐 token 结构,零初始化会把 pose 的 value 也清零 ——
split gate 就成了摆设。现在这个矛盾组合会**直接报错**而不是静默失效。
冷启动完全由 logit bias 承担(−5 约压制 150 倍,且可学习)。

---

## 修订 2 — A3 取消

**A3 与 B4 在目标上互相抵消**:A3 要让 syn 通道在 step 0 是"训练过的接口",
B4 要让它是"彻底的 no-op"。同时开启的话,B4 会在第一步把 A3 训出来的投影全部关掉。

另外实现中发现维度对不上:`GoalPoseEncoder` 输出 3072(FLUX txt 空间),
而 `to_key/to_value` 接收 768。共享投影要把 goal encoder 改成 768,
goal token 彻底离开 txt 空间,连带改 Stage1 的 MoT 调用、注意力 mask、
`collect_trainable_parameters`、bridge 映射。

而收益只有 5 组投影 ≈ 23.6M 参数(占 165M aggregator 的 14%),
且**输入分布完全不同**(Stage1 喂 oracle 位姿 latent,Stage2 喂新初始化聚合器的输出)。

**结论**:先验活在 ActionDiT 里,不活在 syn 投影里。B4 是直接机制,A3 是被它抵消的
间接替代。取消 A3。

---

## 修订 3 — 两个 float mask 的运行期 bug

门控产出的是 float 加性 mask(这是唯一能真正不吃 softmax 质量的做法),
但代码里有两处只考虑了 bool mask:

1. **`return_attn_probs` 路径会崩** —— `scores.masked_fill(~attn_mask, ...)`,
   而 `~float_tensor` 抛
   `TypeError: ~ (operator.invert) is only implemented on integer and Boolean-type tensors`。
   **已实测确认。** 这条路径正是注意力捕获用的,也就是探针 D2/P3 要走的路。
   改成 float mask 时走加法。
2. **SDPA dtype** —— float32 mask 配 bf16 query。**实测这个 PyTorch 版本不报错**,
   我原本断言会炸是错的。cast 保留为防御性:`mot_force_flash_attention` 若被打开,
   flash 后端对 dtype 严格;显式 cast 也避免整个注意力被隐式提升到 float32。

新增 `tests/test_mixed_attention_gate.py`,用真实张量过一遍注意力
(bf16 端到端、强负 bias 等价于硬屏蔽、注意力捕获、bool 路径不回归)——
这两个 bug 在 mask 层面的测试里抓不到。

---

## 接线审计

`scripts/audit_goal_prior_v2.py` —— 从 YAML 构出模型,逐项验证配置真的生效。
**29 项全过**。其中最关键的一条:

> B2+ 的兜底状态与 B4 的 step-0 状态,**留活的恰好是同 8 个 pose 列**。

这是两个机制一致性的机器检查,不是靠注释约定。

测试总数 **52 passed**。

---

## 修订 4 — 可观测性,以及一个会卡死训练的分布式 bug

一次改 6 项而没有消融预算,所以训练过程本身必须可解读。加了两组标量,
都几乎零成本(需要的量本来就有):

| 指标 | 回答什么问题 |
|---|---|
| `train/gate/bias_{mean,first,last}` | **门控活下来了吗?** bias 是可学习的,而 in-dist 训练奖励把它开大。终点若接近 0,说明 syn 通道又退化成了单纯的 augment ——「gate 而非 augment」在终点不成立。这是该主张**唯一的直接证据** |
| `train/loss_action_fallback` vs `train/loss_action_steered` | **退路建起来了吗?** 前者是 blackout 样本(只有 `[txt \| pose \| action]`)的动作 loss。它决定 OOD 时能退回多好的策略 |
| `train/blackout_frac` | 采样正确性的哨兵 |

**顺带发现一个会卡死训练的 bug。** trainer 在
`for key, value in loss_dict.items()` 里逐键调用 `accelerator.gather`。
`loss_action_fallback` 原本只在"该 rank 本步有屏蔽样本"时才存在,而
bs=10、p=0.10 时**单 rank 一步内没有屏蔽样本的概率是 0.9¹⁰ ≈ 35%** ——
不同 rank 的键集不一致 → gather 调用次数不一致 → **NCCL 集合操作错配,挂住**。
8 卡下几乎每步必现。

两处修:

1. **键恒定发出**(子集为空时回落到整体均值)
2. **blackout 改成定量抽取**:每步每 rank 恰好 `k = round(p·B)` 个,
   并保证 `1 ≤ k ≤ B-1`。bs=10、p=0.10 → 恒为 1(实测 1000 步全是 1)。
   两种模式恒共存,指标恒有定义,方差也更小

---

## 最终状态

- 测试 **59 passed**
- 接线审计 **29/29**
- `dark per step (bs=10) = {1}`,`gated_span = (8, 100)`

### 训练时该盯什么

| 时刻 | 看什么 | 判据 |
|---|---|---|
| 开局 1h | `loss_action` | 应远低于旧版的 **0.80**,且领先维持 > 1000 步。不达标 → 补预计算版 L_prior |
| 全程 | `gate/bias_last` | 若快速冲到 0 以上,说明门被完全打开,B2+ 是唯一还在起作用的机制 |
| 全程 | `loss_action_fallback` | 应持续下降并向 `loss_action_steered` 靠拢。若停在高位,说明退路没建起来,OOD 不会改善 |
| 终点 | `loss_pose` | 旧版 raw MSE 0.00137(identity 基线 0.0709)。掉太多说明 blackout 伤到了位姿通道 |

---

# 方案 B —— AE 恢复 ref 通路,firewall 降级为正则化

## 为什么改

全量 LIBERO-Plus(10,030 任务,两模型同一子集,逐 trial 配对):

| 类别 | n | baseline | ours | Δ | p |
|---|---:|---:|---:|---:|---:|
| Robot Initial States | 1550 | 48.58% | **58.65%** | **+10.06** | 0.0000 |
| Background Textures | 1076 | 89.03% | **91.91%** | **+2.88** | 0.0247 |
| Objects Layout | 1525 | 78.49% | 79.34% | +0.85 | 0.393 |
| Light Conditions | 1142 | 98.42% | 96.06% | −2.36 | 0.0007 |
| Sensor Noise | 1601 | 97.31% | 92.19% | −5.12 | 0.0000 |
| Language Instructions | 1537 | 91.74% | 79.64% | −12.10 | 0.0000 |
| Camera Viewpoints | 1599 | 82.93% | 68.29% | −14.63 | 0.0000 |
| **合计** | 10030 | **83.01%** | **79.73%** | **−3.28** | 0.0000 |
| 剔除 Language | 8493 | 81.43% | 79.75% | **−1.68** | **0.0001** |

**注意**:部分数据时"剔除 Language 后不显著"(p=0.11),补满到 8493 后变成 **p=0.0001**。
幅度小了一半但不能说无差异 —— 早期读数会骗人。

**诊断**:ImageWAM 的 baseline 在完全能看像素时 Sensor Noise 97.3%、Light 98.4% ——
它的骨干**本身是去噪器**(`0.5·L_image` 联合训练),ActionDiT 又是 FLUX 权重插值初始化。
**firewall 在解决一个这个宿主上不存在的问题,却拿走了 AE 赖以鲁棒的冗余。**

## 改成什么

```
推理:  [txt | ref(784) | syn(100) | action]     ← 默认,两条通道都在
训练:  都开 0.55 │ 只 ref 0.15 │ 只 syn 0.30
```

两条通道都不可靠 → AE 必须建立冗余。保留 syn-only 状态意味着**同一个 checkpoint
可以切到 firewall 模式**,firewall 从架构约束变成**可评测的能力**:

| 模式 | AE 看到 | 回答 |
|---|---|---|
| Full(默认) | txt + ref + syn | 有没有伤害宿主 |
| Firewall | txt + syn | steering 通道自己扛得动多少 —— thesis 的核心数 |
| Ref-only | txt + ref | 对照 |

参数调整:`syn_gate_bias_init −5 → −2`(key 数 629→1413,分母翻倍,−5 会让 context
的注意力质量掉到 ~0.04%)、`context_token_dropout 0.15 → 0.05`(通道级 dropout
已提供更强的独立性压力,再叠加只会加重训练/推理错配)。`context_blackout_prob 0.10` 保留 ——
它保证 8 个 pose token 承重,否则 92 个自由 latent 会把信息全揽过去,
"pose-based steering" 在实现上就不成立。

---

## Smoke 抓到的两个 bug —— 都只在真实训练里暴露

### ① A2 的 null-image token 是死的

`stage1_null_image_tokens` 既不在 `goal_prior_parameters()`(→ **不在优化器里,永远不更新**)
也不在保存的 payload 里(→ 无法 resume)。配置里写的是"可学习",实际是个冻结的随机常量。
已修三处:优化器、保存、加载(含 fail-closed 白名单)。

### ② 所有 dropout 机制在训练时静默失效 ← 更严重

trainer **只调用 `model.dit.train()` 和 `proprio_encoder.train()`,从不对顶层 ImageWAM
调 `.train()`**,所以 `self.training` 全程是 `False`。我在调用点传了 `training=self.training`,
于是**三态采样、context blackout、逐 token dropout 全部返回 None** ——
配置全对、日志无异常、`gate/bias` 照常记录,但三态和 blackout 指标一个都没有。

**不做 smoke 的话,35 小时训出来的会是"旧版失败配置 + 一条 ref 通路",
所有新机制一个都没生效。**

改为跟随 aggregator 自己的模式(`apply_trainable_policy` 确实把它设成了 train)——
这本来就是这些方法默认值的设计意图,是我在调用点覆盖掉了。

---

## 状态

- 测试 **71 passed**,接线审计 **41/41**
- Stage1 smoke 通过,`stage1_null_image_tokens (32, 128)` 正确落盘
- Stage2 smoke 通过 4 步,验证 ref 拼接 / gate 偏移 / 三态 mask 在真实前向可用

## 训练时的判读

| 指标 | 看什么 |
|---|---|
| `gate/bias_last` | 门有没有被 in-dist 训练开满。终点接近 0 = syn 又变回单纯的 augment |
| `regime/loss_syn_only` | **firewall 模式的实时读数** —— thesis 的核心主张 |
| `regime/loss_ref_only` | 我们有没有伤到宿主原有能力 |
| `loss_action_fallback` | pose-only 兜底的质量 |
| `blackout_frac` / `regime/frac_*` | 采样正确性哨兵 |
