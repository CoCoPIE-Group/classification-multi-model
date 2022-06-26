#!/bin/bash
NUM_PROC=${1:-4}
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train_script_main.py \
  --model resnet18 -b 256 --sched cosine --lr 0.1 --warmup-epochs 5 \
  --weight-decay 1e-4 --reprob 0.4 --remode pixel \
  --config ./configs/dense_resnet18/dense_resnet18.json
