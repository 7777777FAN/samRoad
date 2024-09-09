import os
import numpy as np
import argparse
import shutil
import json 
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from tqdm import tqdm


IMAGE_SIZE = 2048
SEGMENT_WIDTH = 3  # 生成5像素宽路面mask，要跟论文中使用的mask对上，论文samRoad用CV画的，3像素宽度的，一边就有（3+1）/2 = 2条线，加上中间就有5像素宽度
# 不理解为什么设置为5请看OneNote->小点点
# 为了根所有方法对比，设置为3

def create_directory(dir,delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir,exist_ok=True)
    

# 计算指标的函数
def evaluate(args, names):
    def calculate_scores(gt_points,pred_points):
        gt_tree = cKDTree(gt_points)
        if len(pred_points):
            pred_tree = cKDTree(pred_points)
        else:
            return 0,0,0
        thr = args.relax
        dis_gt2pred, _ = pred_tree.query(gt_points, k=1)
        dis_pred2gt, _ = gt_tree.query(pred_points, k=1)
        recall = len([x for x in dis_gt2pred if x < thr]) / len(dis_gt2pred)
        acc = len([x for x in dis_pred2gt if x < thr]) / len(dis_pred2gt)
        r_f = 0
        if acc*recall:
            r_f = 2*recall * acc / (acc+recall)
            
        return acc, recall, r_f


    def pixel_eval_metric(pred_mask,gt_mask):
        def tuple2list(t):
            return [[t[0][x],t[1][x]] for x in range(len(t[0]))]
        
        gt_points = tuple2list(np.where(gt_mask!=0))
        pred_points = tuple2list(np.where(pred_mask!=0))
        # print(len(gt_points))
        # print(len(pred_points))
        
        return calculate_scores(gt_points,pred_points)
    
    
    scores = []
    # for name in os.listdir(f'{args.savedir}/test/skeleton'):
    for name in tqdm(names):
        # pred_graph = np.array(Image.open(f'{args.savedir}/mask/{name}_road.png'))     # 预测的MASK
        pred_graph = np.array(Image.open(f'{args.savedir}/skel_from_graph/{name}.png')) # 由预测图生成的但像素宽度的道路线
        # pred_graph = np.array(Image.open(f'{args.savedir}/mask/{name}_skeleton.png'))     # 由MASK细化得到的skel
        
        # gt_graph = np.array(Image.open(f'./segment_{SEGMENT_WIDTH}/{name}.png'))  # RNGDet的MASK
        # gt_graph = np.array(Image.open(f'/home/godx/research/source_code/sam_road/cityscale/processed/road_mask_{name}.png'))   # samRoad的MASK
        gt_graph = np.array(Image.open(f'/home/space7T/godx/research/samRoad/samRoad/metrics/segment_3/{name}.png'))   # 根据数据集的图数据生成的3像素宽度的mask
        
        # p, r, f1 = pixel_eval_metric(pred_graph>0.364 * 255, gt_graph)
        p, r, f1 = pixel_eval_metric(pred_graph,gt_graph)   # skeleton已经做过二值化了
        scores.append((p, r, f1))
        with open(f"{args.savedir}/results/pixelscore_{args.relax}/{name}.txt", 'w') as f:
            f.write(f"{p:.6f} {r:.6f} {f1:.6f}") 
            
    return round(sum([x[0] for x in scores])/(len(scores)+1e-7),3),\
            round(sum([x[1] for x in scores])/(len(scores)+1e-7),3),\
            round(sum([x[2] for x in scores])/(len(scores)+1e-7),3)



def cal_pixelscore(args):
    create_directory(f"{args.savedir}/results/pixelscore_{args.relax}", delete=True)
    create_directory(f"{args.savedir}/score", delete=True)
    
    print("======================== Calculate Pixel score ========================")
    print(f"Step 1: generating {SEGMENT_WIDTH} pixel wide gt mask")
    # 评价cityscale时用
    names = [8, 9, 19, 28, 29, 39, 48, 49, 59, 68, 69, 79, 88, 89, 99, 108, 109, 119, 128, 129, 139, 148, 149, 159, 168, 169, 179]
    if not os.path.exists(f'./segment_{SEGMENT_WIDTH}'):
        create_directory(f'./segment_{SEGMENT_WIDTH}', delete=False)
        for patch_name in tqdm(names):
            with open(f'/home/space7T/godx/research/my_code/exp_repo/graph/RNGDetPP/cityscale/data/graph/{patch_name}.json','r') as jf:
                edges = json.load(jf)["edges"]
                
            global_mask = Image.fromarray(np.zeros((IMAGE_SIZE,IMAGE_SIZE))).convert('RGB')
            draw = ImageDraw.Draw(global_mask)
            global_mask.load()

            for e in edges:
                for i, v in enumerate(e['vertices'][1:]):
                    draw.line([e['vertices'][i][0],e['vertices'][i][1],v[0],v[1]],width=SEGMENT_WIDTH ,fill=(255,255,255))
            global_mask.save(f'./segment_{SEGMENT_WIDTH}/{patch_name}.png')
        
    print(f"Mask generation finished")   
    print(f"Step 2: calculating pixel scores")
    mean_p, mean_r, mean_f1 = evaluate(args, names)
    with open(f"{args.savedir}/score/pixelscore_{args.relax}.json", 'w') as f:
        f.write(f"{mean_p} {mean_r} {mean_f1}")
    print(f"Calculate Pixel score done!")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # logging config
    parser.add_argument("--savedir", type=str)

    parser.add_argument("--image_size", type=int, default=4096)
    parser.add_argument("--ROI_SIZE", type=int, default=128)
    
    parser.add_argument("--relax", type=int, default=3)
    
    args = parser.parse_args()
    cal_pixelscore(args)