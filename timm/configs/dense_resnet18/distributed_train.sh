#!/bin/bash
NUM_PROC=${1:-4}
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train_script_main.py \
  --config ./configs/dense_resnet18/args_ai_template.json