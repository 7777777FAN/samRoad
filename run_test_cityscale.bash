#!/bin/bash
python inferencer.py \
 --config=config/toponet_vitb_512_cityscale.yaml \
 --checkpoint=./lightning_logs/GTE加了噪声并调整损失权重/epoch=37-step=95000.ckpt