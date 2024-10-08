#!/bin/bash
# decode, 计算APLS和可视化预测结果

output_dir='./save/可学习decoder_23epoch' 

# decode GTE_logits outputed by model
python for_decode.py --output_dir="$output_dir" --config=config/toponet_vitb_512_cityscale.yaml

# cal metrcis(apls)
cd ./metrics
./apls_for_samGTE.bash $output_dir no_simplify

# VIS
cd .. 
python vis_pred_graph.py  --output_dir="$output_dir" \

    # --simplify 