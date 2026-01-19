# Consistency Flow Matching SFT + PPO + RFC

## 方法概述

两阶段训练 Pi0.5 机器人策略：
1. **阶段一**：Consistency Flow Matching (CFM) + EMA 离线监督学习 —— **SFT阶段拉直轨迹**
2. **阶段二**：PPO 在线强化学习 + RFC 继续保持直线轨迹

---

## 阶段一：Consistency SFT (CFM + EMA)

### 核心思想

在SFT阶段就把flow轨迹拉直，而不是等到RL阶段。使用**EMA Teacher + Student**的自蒸馏方式：

```
Student (θ) ←── 学习 ──→ EMA Teacher (θ_ema)
     ↓                           ↓
  预测 v_θ(x_t, t)          生成配对 (x_0, x_1)
```

### 训练流程

每个训练step：

1. **EMA Teacher 生成轨迹配对**
   - 采样噪声 $x_1 \sim \mathcal{N}(0, I)$
   - 用EMA模型多步去噪得到 $x_0^{ema}$
   - 得到配对 $(x_0^{ema}, x_1)$

2. **Student 学习直线速度**
   - 随机采样 $t \sim U[\delta_{min}, \delta_{max}]$
   - 构造 $x_t = (1-t) \cdot x_0^{ema} + t \cdot x_1$
   - 目标：$v_\theta(x_t, t) \to (x_1 - x_0^{ema})$

3. **总Loss**
   $$\mathcal{L} = (1-\lambda) \cdot \mathcal{L}_{FM} + \lambda \cdot \mathcal{L}_{CFM}$$

   - $\mathcal{L}_{FM}$：标准Flow Matching loss（用GT action）
   - $\mathcal{L}_{CFM}$：Consistency loss（用EMA生成的配对）
   - $\lambda = 0.25$

4. **更新EMA Teacher**
   $$\theta_{ema} \leftarrow \alpha \cdot \theta_{ema} + (1-\alpha) \cdot \theta$$
   - $\alpha = 0.995$

### 关键配置

```yaml
actor:
  model:
    openpi:
      cfm_enabled: true
      cfm_loss_weight: 0.25      # 25% CFM + 75% FM
      cfm_ema_enabled: true
      cfm_ema_decay: 0.995       # EMA衰减率
      cfm_delta_t_min: 0.01      # t采样下界
      cfm_delta_t_max: 0.1       # t采样上界
```

### 为什么EMA有效？

| 无EMA | 有EMA |
|-------|-------|
| 模型自己生成target，容易collapse | EMA提供稳定的target |
| 训练不稳定 | 平滑的自蒸馏 |
| 容易过拟合 | 更好的泛化 |

---

## 阶段二：PPO + RFC 在线微调

### 目的

1. **PPO**：通过RL优化策略，提升任务成功率
2. **RFC**：继续保持直线轨迹，防止RL训练破坏CFM学到的直线性

### RFC Loss

对于在线rollout得到的 $(x_0, x_1)$ 配对：

$$\mathcal{L}_{RFC} = \mathbb{E}_{t \sim U(0,1)} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]$$

### Reflow Mask

控制哪些样本参与RFC loss计算，通过`reflow_mask_type`配置：

| 策略 | 描述 |
|------|------|
| `success_only` | 只有成功episode参与（默认） |
| `advantage_positive` | advantage > 0 的样本参与 |
| `advantage_topk` | advantage排名前K%的样本参与 |

```yaml
algorithm:
  rfc_enabled: True
  rfc_lambda: 0.05
  reflow_mask_type: "advantage_positive"  # 或 success_only, advantage_topk
  reflow_topk_ratio: 0.5                  # 仅用于 advantage_topk
```

**设计动机**：不是所有rollout数据都适合做reflow。`advantage_positive`让模型只从"好"的动作学习直线轨迹。

### 总训练目标

$$\mathcal{L}_{total} = \mathcal{L}_{PPO} + \mathcal{L}_{critic} + \lambda_{RFC} \cdot \mathcal{L}_{RFC}$$

- $\lambda_{RFC}$：0.05-0.1

---

## 整体Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  阶段一：Consistency SFT                                         │
│  ┌──────────┐    EMA更新    ┌──────────┐                        │
│  │ Student  │ ────────────→ │  Teacher │                        │
│  │    θ     │               │   θ_ema  │                        │
│  └────┬─────┘               └────┬─────┘                        │
│       │                          │                              │
│       │ FM Loss                  │ 生成 (x_0, x_1)              │
│       │ (GT action)              │                              │
│       ↓                          ↓                              │
│  ┌─────────────────────────────────────┐                        │
│  │  L = 0.75·L_FM + 0.25·L_CFM         │ ← 拉直轨迹             │
│  └─────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  阶段二：PPO + RFC                                               │
│  ┌──────────┐    在线rollout   ┌─────────┐                      │
│  │  Policy  │ ───────────────→ │   Env   │                      │
│  │    θ     │ ←── reward ───── │ LIBERO  │                      │
│  └────┬─────┘                  └─────────┘                      │
│       │                                                         │
│       ↓                                                         │
│  ┌─────────────────────────────────────┐                        │
│  │ L = L_PPO + L_critic + λ·L_RFC      │ ← 保持直线性           │
│  └─────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 实验结果 (LIBERO-Spatial)

| 方法 | 去噪步数 | 成功率 (ID) | 成功率 (OOD) |
|------|---------|-------------|--------------|
| Base (Pi0.5 SFT) | 5 | 79.5% | 19.3% |
| CFM+EMA 1000步 | 2 | 85.8% | 20.5% |
| CFM+EMA 500步 | 2 | 77.0% | **21.6%** |
| PPO+RFC 80步 | 2 | 88.25% | - |
| PPO v9 160步 | 4 | **90.0%** | - |

### 关键发现

1. **CFM+EMA在SFT阶段就能2步去噪**：85.8% vs Base的79.5%
2. **EMA防止过拟合**：OOD泛化更好（21.6% vs 18.2%无EMA）
3. **PPO+RFC进一步提升ID性能**：88.25%

---

## 参考

- [Consistency Models](https://arxiv.org/abs/2303.01469)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
