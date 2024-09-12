# 可视化推理结果
import json
import pickle
import cv2 as cv
from PIL import Image
import os, shutil
import os.path as osp

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--simplify', default=False, action='store_true',
                    help='visulize simplified graph outputed by douglaspeucker algrithm or original dense graph')

parser.add_argument('--output_dir', type=str)


if '__main__' == __name__:

    args = parser.parse_args()
    
    sat_pattern = './cityscale/20cities/region_{}_sat.png'
    gt_graph_pattern = './cityscale/20cities/region_{}_graph_gt.pickle'
    if args.simplify:
        pred_graph_pattern = osp.join(args.output_dir,'decode_result/region_{}_graph_simplified.p')
    else:
        pred_graph_pattern = osp.join(args.output_dir,'decode_result/region_{}_graph_nosimplify.p')

    save_dir = osp.join(args.output_dir, 'vis_pred_graph')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    save_pattern = os.path.join(save_dir, 'region_{}_vis.png')

    test_idxs = json.load(open('./cityscale/data_split.json', 'rb'))['test']
    # test_idxs = [49, 179]

    for idx in test_idxs:
        sat_path = sat_pattern.format(idx)
        pred_graph_path = pred_graph_pattern.format(idx)
        
        sat = cv.imread(sat_path)
        pred_graph = pickle.load(open(pred_graph_path, 'rb'))
        
        for n, v in pred_graph.items():
            src_y, src_x = n
            for nei in v:
                dst_y, dst_x = nei
                cv.line(sat, (src_x, src_y), (dst_x, dst_y), color=(15, 160, 253), thickness=4)
                cv.circle(sat, (dst_x, dst_y), radius=4, color=(0, 255, 255), thickness=-1)
            cv.circle(sat, (src_x, src_y), radius=4, color=(0, 255, 255), thickness=-1)
        
        save_path = save_pattern.format(idx)
        cv.imwrite(save_path, sat)