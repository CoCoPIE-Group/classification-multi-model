#!/bin/bash
NUM_PROC=${1:-4}
LR=${2:-0.048}
uniform=${3:-u}
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train_script_main.py \
  --model efficientnet_b0 -b 384 --sched step --decay-epochs 2.4 --decay-rate .97 \
  --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 \
  --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr $lr \
  --config ./configs/dense_effnetb0/dense_effnetb0.json
