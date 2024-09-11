#!/bin/bash
python inferencer.py \
 --config=config/toponet_vitb_512_cityscale.yaml \
 --checkpoint=./lightning_logs/GTE加了噪声/epoch=59-step=150000.ckpt