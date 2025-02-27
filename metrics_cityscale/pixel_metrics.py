import numpy as np
from PIL import Image
import os
import shutil
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import argparse
import cv2 as cv
import pickle



class Evaluator(object):
    '''基于混淆矩阵给出评估指标, 输入应该是对应的分类结果, 也即gt和pred都是0, 1矩阵'''
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Pixel_Precision(self):
        self.precision = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[0, 1])
        return self.precision

    def Pixel_Recall(self):
        self.recall = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[1, 0])
        return self.recall

    def Pixel_F1(self):
        f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        return f1

    def Intersection_over_Union(self):
        IoU = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[1, 0] + self.confusion_matrix[0, 1] + 1e-10)
        return IoU

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def create_directory(dir,delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir,exist_ok=True)


def pred_graph_drawer(patch_name, args):
    pred_graph = pickle.load(open(f"../{args.savedir}/graph/{patch_name}.p", 'rb'))
    
    save_path = f'../{args.savedir}/roadmap/{patch_name}.png'
    if os.path.exists(save_path):
        return
    
    pred_graph_skeleton = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for n, v in pred_graph.items():
        src_y, src_x = n
        for nei in v:
            dst_y, dst_x = nei
            cv.line(pred_graph_skeleton, (src_x, src_y), (dst_x, dst_y), color=255, thickness=3)
    cv.imwrite(save_path, pred_graph_skeleton)
      
    
def cal_pixelscores(evaluator, name):
    gt_path = os.path.join('../cityscale/processed_3_width/', f'road_mask_{name}.png')
    pred_path = os.path.join(f'../{args.savedir}/roadmap', f'{name}.png')
    
    gt_mask = np.array(Image.open(gt_path)).astype(np.uint8)
    pred_mask = np.array(Image.open(pred_path)).astype(np.uint8)
    
    gt_mask = gt_mask != 0
    pred_mask = pred_mask != 0
    
    evaluator.add_batch(gt_mask, pred_mask)





if '__main__' == __name__:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str)
    parser.add_argument("--image_size", type=int, default=2048)
    parser.add_argument("--trace", type=lambda x: x.lower()=='true', default=False)
    
    args = parser.parse_args()
    IMAGE_SIZE = args.image_size
    
    create_directory(f'../{args.savedir}/roadmap', delete=True)
    
    names = [8, 9, 19, 28, 29, 39, 48, 49, 59, 68, 69, 79, 88, 89, 99, 108, 109, 119, 128, 129, 139, 148, 149, 159, 168, 169, 179]
    
    print("======================== Calculate Pixel score ========================")
    print(f"===> Step 1: generating binary road map for predicted graph")
    pool = Pool()
    graph_drawer_with_args = partial(pred_graph_drawer, args=args)
    pool.map(graph_drawer_with_args, names)
    pool.close()
    pool.join()
    print(f"<=== Binary road map  generation finished")
    
    print(f"===> Step 2: calculating pixel scores")
    evaluator = Evaluator(2)
    evaluator.reset()
    for name in tqdm(names):
        cal_pixelscores(evaluator, name)
    pre, rec, f1, iou = evaluator.Pixel_Precision(), evaluator.Pixel_Recall(), evaluator.Pixel_F1(), evaluator.Intersection_over_Union()
    
    if not os.path.exists(f'../{args.savedir}/score'):
        os.makedirs(f'../{args.savedir}/score')
    with open(f"../{args.savedir}/score/pixelscore.json", 'w') as f:
        f.write(f"pre: {pre}, rec: {rec}, f1: {f1}, IOU: {iou}")