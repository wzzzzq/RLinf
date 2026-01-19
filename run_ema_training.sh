#!/bin/bash
# EMA CFM training script for Pi0.5 on Libero

set -e

# 1. GPU配置 - 使用GPU 4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 2. Ray隔离 - 使用本地Ray
export RAY_ADDRESS=local

# 3. 环境变量
export WANDB_API_KEY=489fe7b734df1e91930d434d63c36b600b2faed9
export EMBODIED_PATH=/mnt/pfs/scalelab/ziqian/RLinf/examples/embodiment
export HF_LEROBOT_HOME=/mnt/pfs/scalelab/hf_cache/lerobot
export PYTHONPATH=/mnt/pfs/scalelab/ziqian/RLinf:$PYTHONPATH

# 4. Python环境
PYTHON=/mnt/pfs/scalelab/pi05_venv/bin/python

# 5. 创建日志目录
LOG_DIR=/mnt/pfs/scalelab/ziqian/RLinf/logs/$(date +'%Y%m%d-%H%M%S')_cfm_ema_sft_pi05
mkdir -p $LOG_DIR

echo "Starting CFM + EMA training for Pi0.5 on Libero..."
echo "Log directory: $LOG_DIR"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "EMA enabled with decay=0.995"

# 6. 启动训练
cd /mnt/pfs/scalelab/ziqian/RLinf
$PYTHON examples/sft/train_embodied_sft.py \
    --config-name=libero_fm_sft_pi05 \
    runner.logger.log_path=$LOG_DIR \
    runner.logger.experiment_name="cfm_sft_pi05_libero_ema" \
    2>&1 | tee $LOG_DIR/training.log
