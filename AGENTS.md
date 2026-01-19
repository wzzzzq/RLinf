# RLinf Pi0.5 Training

## python环境
使用这个venv /mnt/pfs/scalelab2/pi05_venv/bin/python

## 路径说明
- **A800-1**: `/mnt/pfs/scalelab/ziqian/RLinf`
- **A800-2**: `/mnt/pfs/scalelab2/ziqian/RLinf`
- **4090**: /pfs/pfs-ilWc5D/ziqian/RLinf
- **注意**: A800-1上把`scalelab2`换成`scalelab`

## 代理说明
访问github,huggingface 等需要代理
export http_proxy=http://172.16.0.136:18000 
export https_proxy=http://172.16.0.136:18000

## 环境配置
```bash
export WANDB_API_KEY=489fe7b734df1e91930d434d63c36b600b2faed9
```

## 启动训练前必做：重定向Ray

```bash
# 设置Ray临时目录和容量阈值（避免/tmp空间不足）
export RAY_local_fs_capacity_threshold=0.99
export RAY_TMPDIR="/mnt/pfs/scalelab2/ch/RLinf/ray_tmp"
```

启动训练前，务必确认路径正确，比如如果在A800-1, scalelab2统一要改成scalelab再启动训练

---
important: kill掉任何之前进程必须先询问

# 训练启动

使用 `training-launcher-monitor` agent 启动训练。详细配置和启动参数参考 agent 文档。
不要直接启动训练，通过agent启动tmux窗口内启动训练

关键信息：
- **Config**: `examples/sft/config/libero_fm_sft_pi05.yaml`
- **Base model**: `/mnt/pfs/scalelab/ziqianwang/models/RLinf-Pi05-SFT`

---

# 评测

使用 `embodied-eval-monitor` agent 启动评测。详细配置参考 agent 文档。

## LIBERO-Spatial 评测
- **Config**: `examples/embodiment/config/libero_spatial_eval_cfm.yaml`

## LIBERO-90 泛化评测
- **Config**: `examples/embodiment/config/libero_90_eval_cfm.yaml`
- 90个未见过的长程任务，450个环境

---

# 评测结果汇总 (temp=0, success_once)

## 完整对比表格

| 模型 | 去噪步数 | LIBERO-Spatial (ID) | LIBERO-90 (OOD) |
|------|---------|---------------------|-----------------|
| **PPO v9 160步** | 4 | **90.0%** | - |
| PPO CFM noise03 80步 | 2 | 88.25% | - |
| CFM+EMA 2GPU 1000步 | 1 | 87.3% | 15.9% |
| PPO Flow Noise v2 40步 | 2 | 82.75% | - |
| CFM+EMA 2GPU 1000步 | 2 | 85.8% | 20.5% |
| Base (Pi0.5 SFT) | 5 | 79.5% | 19.3% |
| PPO v9 80步 | 4 | 77.5% | - |
| CFM+EMA 2GPU 500步 | 2 | 77.5% | **21.6%** |
| CFM+EMA 500步 | 2 | 77.0% | **21.6%** |
| CFM (无EMA) 500步 | 2 | - | 18.2% |
| CFM (无EMA) 1000步 | 2 | - | 18.2% |

## 关键发现

1. **2步去噪是最佳平衡点**：在ID和OOD上都有较好表现
2. **PPO CFM noise03 80步接近最佳**：88.25% 接近 PPO v9 160步的 90%
3. **EMA有效防止过拟合**：CFM+EMA在OOD上比无EMA版本更稳定

## Checkpoints

| 模型 | 路径 |
|------|------|
| Base (Pi0.5 SFT) | `/mnt/pfs/scalelab/ziqianwang/models/RLinf-Pi05-SFT` |
| PPO v9 160步 | `logs/20260104-153724_ppo_cfm_v9_noise02/ppo_cfm_v9_noise02/checkpoints/global_step_160/actor/model` |
| PPO v9 80步 | `logs/20260104-153724_ppo_cfm_v9_noise02/ppo_cfm_v9_noise02/checkpoints/global_step_80/actor/model` |
| PPO Flow Noise v2 40步 | `logs/20260109-163035_ppo_flow_noise_v2/ppo_flow_noise_v2/checkpoints/global_step_40/actor/model` |
| CFM (无EMA) 1000步 | `/mnt/pfs/scalelab/ziqian/results/cfm_sft_pi05_libero/checkpoints/global_step_1000/actor/model` |
| CFM+EMA 500步 | `logs/20251230-153847_cfm_ema/cfm_sft_pi05_ema/checkpoints/global_step_500/actor/model` |
| CFM+EMA 2GPU 500步 | `logs/20251230-215715_cfm_ema_2gpu/cfm_sft_pi05_ema_2gpu/checkpoints/global_step_500/actor/model` |
| CFM+EMA 2GPU 1000步 | `logs/20251230-215715_cfm_ema_2gpu/cfm_sft_pi05_ema_2gpu/checkpoints/global_step_1000/actor/model` |

## WandB

- CFM (无EMA): `christianwang-sjtu/rlinf/qga37t8h`

---

# ManiSkill PPO 训练

使用 `training-launcher-monitor` agent 启动训练。详细配置参考 agent 文档。

## 模型路径
- **SFT Base**: `/mnt/pfs/scalelab/ziqian/models/RLinf-Pi05-ManiSkill-25Main-SFT`
- **Config**: `examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml`

---

# Reflow Mask 配置

RFC loss支持多种mask策略，控制哪些样本参与reflow训练。

## 配置项

```yaml
algorithm:
  rfc_enabled: True
  rfc_lambda: 0.05
  reflow_mask_type: "success_only"  # 可选: success_only, advantage_positive, advantage_topk
  reflow_topk_ratio: 0.5            # 仅用于 advantage_topk
```

## 策略说明

| 策略 | 描述 |
|------|------|
| `success_only` | 只有成功episode参与（默认，原有行为） |
| `advantage_positive` | advantage > 0 的样本参与 |
| `advantage_topk` | advantage排名前K%的样本参与 |

**注意**: mask粒度是sample级别，每个sample整体参与或不参与RFC loss。

## 配置文件

| 配置 | GPU | 策略 |
|------|-----|------|
| `libero_spatial_ppo_rfc.yaml` | 0,1,2,3 | success_only |
| `libero_spatial_ppo_rfc_adv_positive.yaml` | 4,5,6,7 | advantage_positive |

## 监控指标

WandB中查看 `actor/reflow_mask_ratio` 显示参与RFC loss的样本比例。

---

# DSRL (Diffusion Steering via RL)

用 PPO 训练 Latent Actor 预测初始噪声，替代随机噪声采样。

## 核心思想

```
传统 Flow Matching:  random noise → Pi0.5 denoise → action
DSRL:               Latent Actor(s) → noise → Pi0.5 denoise → action
                         ↑
                    PPO 训练 (最大化 reward)
```

- **Latent Actor**: MLP 网络，输入 `prefix_output.mean + state`，输出噪声分布 `N(μ, σ)`
- **Pi0.5 Action Expert**: 完全冻结，只做推理
- **可选配合 RFC**: 用成功数据做 reflow 微调 action expert

## 配置文件

| 配置 | 描述 |
|------|------|
| `libero_spatial_dsrl_only.yaml` | DSRL only，与官方 PPO 配置一致 |
| `libero_spatial_dsrl.yaml` | DSRL + RFC |

## 关键配置项

```yaml
algorithm:
  dsrl_enabled: True   # 启用 DSRL
  rfc_enabled: False   # 是否同时启用 RFC

actor:
  model:
    openpi:
      dsrl_enabled: True
      dsrl_state_dim: 2056      # 2048 (prefix) + 8 (state)
      dsrl_hidden_dims: [512, 256, 128]
```

## 新增文件

| 文件 | 描述 |
|------|------|
| `rlinf/models/embodiment/modules/latent_actor.py` | Latent Actor 网络 |
| `examples/embodiment/config/libero_spatial_dsrl_only.yaml` | DSRL 配置 |
| `examples/embodiment/config/libero_spatial_dsrl.yaml` | DSRL + RFC 配置 |

