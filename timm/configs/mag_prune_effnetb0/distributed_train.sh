#!/bin/bash
NUM_PROC=${1:-4}
EPOCHS=${2:-300}
LR=${3:-0.0048}
uniform=${4:-u}
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train_script_main.py /data/imagenet --model efficientnet_b0 -b 384 \
   --sched step --epochs $EPOCHS --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 \
   --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 --aa rand-m9-mstd0.5 \
   --remode pixel --reprob 0.2 --amp --lr $LR --pretrained --log-wandb --config \
   configs/mag_prune_effnetb0/mag_prune_${uniform}0.6_effnetb0.json --warmup-epochs 0

