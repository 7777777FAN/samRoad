#!/bin/bash

python train.py --config=config/toponet_vitb_512_cityscale.yaml  \
    --resume ./lightning_logs/修正损失计算方式/checkpoints/epoch=39-step=100000.ckpt