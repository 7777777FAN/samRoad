# 可以初步可视化GTE以及完全解码GTE
import cv2 as cv
from skimage import measure
from scipy.ndimage import rotate
from tqdm import tqdm
import numpy as np
import os.path as osp
import os
import torch
import shutil
import json

from argparse import ArgumentParser

from decoder import DecodeAndVis

IMG_SIZE = 2048
OFFSET = 1
MAX_DEGREE = 6
VECTOR_NORM = 25.0

NUM_PROCESS = 10
NUM_THREAD = 10
 
 
parser = ArgumentParser()
parser.add_argument('--output_dir', default=None, type=str)



def create_dir(dir, exist_ok=True):
    if os.path.exists(dir):
        if exist_ok:
            return
        else:
            shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=False)


def vis_GT_GTE(GTE, verify_dir, keypoint_thr=0.1, edge_thr=0.1, aug=False, rot_angle=90, rot_index=0):
    vis_output = np.zeros((IMG_SIZE, IMG_SIZE, 3))    # 不加底图
    # vis_output = cv.imread(rgb_pattern.format(7))[:, :, :]
    sub_GTE = torch.tensor(GTE[:, :, :], dtype=torch.float32)
    
    if aug:
        sub_GTE = GTE[:, :, :]
        rot_mat = np.array([    # 只针对90度的旋转，如果是任意角度还需要改旋转矩阵
                        [0, 1],
                        [-1, 0]
            ], dtype=np.float32)

        sub_GTE = rotate(sub_GTE, rot_angle*rot_index, axes=(0, 1), reshape=False)
        vis_output = rotate(vis_output, rot_angle*rot_index, axes=(0, 1), reshape=False)
        
        # 只对所有不为0的进行变换，其余的全是0，没必要再计算了
        for r, c in np.column_stack(np.where(sub_GTE[:,:, 0]>0.98)):
            delta_coords = []
            for j in range(MAX_DEGREE):
                delta_coords.append([sub_GTE[r, c, 1+3*j+1], sub_GTE[r, c, 1+3*j+2]])   # dx, dy
            delta_coords_to_rot = np.column_stack(delta_coords)
            roted_coords = np.linalg.matrix_power(rot_mat, rot_index)@delta_coords_to_rot
            
            for j in range(MAX_DEGREE):    # stick back
                sub_GTE[r, c, 1+3*j+1], sub_GTE[r, c, 1+3*j+2] = roted_coords[:, j]
                
    keypoint_map = sub_GTE[:,:, 0].detach().cpu().numpy()

    # 从团中寻找中心点
    keypoint_map = (keypoint_map > keypoint_thr).astype(np.uint8)
    # cv.imwrite(osp.join(verify_dir, f'region_8_vis_{rot_angle*rot_index}.png'), vis_output)
    # cv.imwrite(osp.join(verify_dir, f'region_8_vis_{rot_angle*rot_index}.png'), keypoint_map*255)
    labels = measure.label(keypoint_map, connectivity=2)
    props = measure.regionprops(labels)
    min_area = 0
    for region in props:
        if region.area > min_area:
            center = region.centroid[::-1]   # rc-> xy
            center_x, center_y = int(center[0]), int(center[1])
            cv.circle(vis_output, (center_x, center_y), radius=2, color=(0, 0, 255), thickness=-1)
            
            r, c = center_y, center_x
            for j in range(MAX_DEGREE):
                edgeness = torch.sigmoid(sub_GTE[r, c, 1+3*j])
                if edgeness > edge_thr:
                    dx, dy = sub_GTE[r, c, 1+3*j+1], sub_GTE[r, c, 1+3*j+2]
                    # dx, dy = rot_mat@np.array([dx, dy])
                    dst_x, dst_y = int(center_x + VECTOR_NORM*dx), int(center_y + VECTOR_NORM*dy)
                    cv.line(vis_output, (center_x, center_y), (dst_x, dst_y), color=(0, 255, 0), thickness=1)
    cv.imwrite(osp.join(verify_dir, f'region_8_vis_{rot_angle*rot_index}.png'), vis_output)
    



if '__main__' == __name__:
    args = parser.parse_args()
    
    rgb_pattern = './cityscale/20cities/region_{}_sat.png'
    GTE_logits_pattern = osp.join(args.output_dir,'GTE_logits/region_{}_GTE_logits.npz')
    data_split_path = './cityscale/data_split.json'
    output_result_dir = osp.join(args.output_dir, 'decode_result')
    output_score_dir = osp.join(output_result_dir, 'score')

    create_dir(output_result_dir, exist_ok=False)
    create_dir(output_score_dir, exist_ok=False)

    test_tile_idxes = json.load(open(data_split_path, 'rb'))['test']

    # test_tile_idxes = [49, 179]
    # test_tile_idxes = [8]
    for idx in tqdm(test_tile_idxes):
        GTE = np.load(open(GTE_logits_pattern.format(idx), 'rb'))['GTE_logits']

        # 可视化，用于检验自己的编码是否正确以及预测结果是否正确（暂不论结果好坏）
        # verify_dir = './save/GTE_加了噪声/verify'
        # if osp.exists(verify_dir):
        #     shutil.rmtree(verify_dir)
        # os.makedirs(verify_dir)
        # vis_GT_GTE(GTE, verify_dir=verify_dir, aug=False)

        # 用Sat2Graph的解码算法解码
        output_file = os.path.join(output_result_dir, f"region_{idx}")
        DecodeAndVis(GTE, output_file, thr=0.05, edge_thr=0.05, angledistance_weight=10, snap=True, imagesize=2048)
        # DecodeAndVis(GTE, output_file, thr=0.05, edge_thr=0.05, angledistance_weight=50, snap=True, imagesize=2048)