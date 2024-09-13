#!/bin/bash
python inferencer.py \
 --config=config/toponet_vitb_512_cityscale.yaml \
 --checkpoint=./lightning_logs/samRoad预测等间距点/checkpoints/epoch=9-step=25000.ckpt