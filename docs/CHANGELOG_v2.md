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
