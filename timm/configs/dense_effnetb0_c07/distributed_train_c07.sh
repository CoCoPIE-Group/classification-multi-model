#!/bin/bash
NUM_PROC=${1:-4}
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train_script_main.py \
  --config ./configs/dense_effnetb0_c07/dense_effnetb0_c07.json


