#!/bin/bash
NUM_PROC=${1:-4}
EPOCHS=${2:-300}
LR=${3:-0.048}
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train_pruning.py /data/imagenet --model efficientnet_b0 -b 384 \
   --sched step --epochs $EPOCHS --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 \
   --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 --aa rand-m9-mstd0.5 \
   --remode pixel --reprob 0.2 --amp --lr $LR --pretrained --log-wandb \
   --pruning_ratio=0.36 --prune --prune_skip --gamma_knowledge=20

