#!/bin/bash
NUM_PROC=${1:-4}
EPOCHS=${2:-300}
LR=${3:-0.0048}
uniform=${4:-u}
config=${5:-admm}

admm_config=configs/admm_prune_effnetb0/admm_prune_${uniform}0.6_effnetb0.json
retrain_config=configs/admm_prune_effnetb0/retrain_prune_${uniform}0.6_effnetb0.json

if [[ $config = admm ]]
  then
    config_prune=${admm_config}
elif [[ $config = retrain ]]
  then
    config_prune=$retrain_config
fi
echo $config_prune

shift

if [[ $config = admm ]]
  then
    python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train_script_main.py /data/imagenet --model efficientnet_b0 -b 384 \
       --sched step --epochs $EPOCHS --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 \
       --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 --aa rand-m9-mstd0.5 \
       --remode pixel --reprob 0.2 --amp --lr $LR --pretrained --log-wandb --config \
       $admm_config --warmup-epochs 0
elif [[ $config = retrain ]]
  then
    python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train_script_main.py /data/imagenet --model efficientnet_b0 -b 384 \
       --sched step --epochs $EPOCHS --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 \
       --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 --aa rand-m9-mstd0.5 \
       --remode pixel --reprob 0.2 --amp --lr $LR --log-wandb \
       --config $retrain_config --warmup-epochs 0 \
       --initial-checkpoint output/train/20220610-051824-efficientnet_b0-224/last.pth.tar
fi
echo $config_prune

