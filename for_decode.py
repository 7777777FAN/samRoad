# %%
# 可视化GTE
import cv2 as cv
from skimage import measure
from scipy.ndimage import rotate
import numpy as np
import os.path as osp
import os
import torch
import shutil



IMG_SIZE = 2048
OFFSET = 1
MAX_DEGREE = 6
VECTOR_NORM = 25.0

NUM_PROCESS = 10
NUM_THREAD = 10
 
# 将图编码为19维张量
rgb_pattern   = './cityscale/20cities/region_{}_sat.png'
GTE_logits_pattern = './save/成功的GTE/GTE_logits/region_{}_GTE_logits.npz'



def vis_GT_GTE(GTE, keypoint_thr=0.1, edge_thr=0., aug=False, rot_angle=90, rot_index=0):
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
    


GTE = np.load(open(GTE_logits_pattern.format(8), 'rb'))['GTE_logits']
verify_dir = './save/成功的GTE/verify'
# if osp.exists(verify_dir):
#     shutil.rmtree(verify_dir)
# os.makedirs(verify_dir)
# vis_GT_GTE(GTE, aug=False)

# %%
# 用Sat2Graph的解码算法解码一下
from decoder import DecodeAndVis

output_file = './save/成功的GTE/decode_result/region_8'
DecodeAndVis(GTE, output_file, thr=0.01, edge_thr=0.1, angledistance_weight=50, snap=True, imagesize=2048)
