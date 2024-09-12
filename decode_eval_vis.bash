#!/bin/bash
# decode, 计算APLS和可视化预测结果

output_dir='./save/GTE_加了噪声并调整损失权重_37epoch' 

# decode GTE_logits outputed by model
python for_decode.py --output_dir="$output_dir"

# cal metrcis(apls)
cd ./metrics
./apls_for_samGTE.bash $output_dir simplify

# VIS
cd .. 
python vis_pred_graph.py --simplify --output_dir="$output_dir"