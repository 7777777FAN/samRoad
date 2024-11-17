#!/bin/bash

python inferencer.py \
 --config=config/toponet_vitb_256_spacenet.yaml \
 --checkpoint=./lightning_logs/可学习decoder/epoch=39-step=100000.ckpt