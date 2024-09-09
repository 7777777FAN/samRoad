#!/bin/bash
python inferencer.py \
 --config=config/toponet_vitb_512_cityscale.yaml \
 --checkpoint=./lightning_logs/修正损失计算方式/checkpoints/epoch=37-step=95000.ckpt