import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
import graph_utils
import rtree
import scipy
import pickle
import os
import addict
import json
from tqdm import tqdm


def read_rgb_img(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def cityscale_data_partition():
    # dataset partition
    indrange_train = []
    indrange_test = []
    indrange_validation = []

    for x in range(180):
        if x % 10 < 8 :
            indrange_train.append(x)

        if x % 10 == 9:
            indrange_test.append(x)

        if x % 20 == 18:
            indrange_validation.append(x)

        if x % 20 == 8:
            indrange_test.append(x)
    return indrange_train, indrange_validation, indrange_test


def spacenet_data_partition():
    # dataset partition
    with open('./spacenet/data_split.json','r') as jf:
        data_list = json.load(jf)
        # data_list = data_list['test'] + data_list['validation'] + data_list['train']
    # train_list = [tile_index for _, tile_index in data_list['train']]
    # val_list = [tile_index for _, tile_index in data_list['validation']]
    # test_list = [tile_index for _, tile_index in data_list['test']]
    train_list = data_list['train']
    val_list = data_list['validation']
    test_list = data_list['test']
    return train_list, val_list, test_list


def get_patch_info_one_img(image_index, image_size, sample_margin, patch_size, patches_per_edge):
    '''获取对一张大影像的裁剪信息, 具体为小patch左上、右下顶点坐标'''
    patch_info = []
    sample_min = sample_margin
    sample_max = image_size - (patch_size + sample_margin)
    eval_samples = np.linspace(start=sample_min, stop=sample_max, num=patches_per_edge)
    eval_samples = [round(x) for x in eval_samples]     # round不传递位数时取整
    for x in eval_samples:
        for y in eval_samples:
            patch_info.append(
                (image_index, (x, y), (x + patch_size, y + patch_size))
            )
    return patch_info


class GraphLabelGenerator():
    def __init__(self, config, full_graph, coord_transform):
        self.config = config
        # full_graph: sat2graph format
        # coord_transform: lambda, [N, 2] array -> [N, 2] array
        # convert to igraph for high performance
        self.full_graph_origin = graph_utils.igraph_from_adj_dict(full_graph, coord_transform)
        # find crossover points, we'll avoid predicting these as keypoints
        self.crossover_points = graph_utils.find_crossover_points(self.full_graph_origin)
        # subdivide version
        # TODO: check proper resolution
        self.subdivide_resolution = 4
        self.full_graph_subdivide = graph_utils.subdivide_graph(self.full_graph_origin, self.subdivide_resolution) # 将每条边分成4段，密集化
        # np array, maybe faster

        self.subdivide_points = np.array(self.full_graph_subdivide.vs['point'])
        # pre-build spatial index
        # rtree for box queries
        self.graph_rtee = rtree.index.Index()
        for i, v in enumerate(self.subdivide_points):
            x, y = v
            # hack to insert single points
            self.graph_rtee.insert(i, (x, y, x, y))
        # kdtree for spherical query
        self.graph_kdtree = scipy.spatial.KDTree(self.subdivide_points)

        # pre-exclude points near crossover points
        crossover_exclude_radius = 4
        exclude_indices = set()
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(p, crossover_exclude_radius)    # 查找crossover点周围4个像素的所有点
            exclude_indices.update(nearby_indices)
        self.exclude_indices = exclude_indices  # 把这些点的索引记下来，后续需要排除在外

        # Find intersection points, these will always be kept in nms
        itsc_indices = set()
        point_num = len(self.full_graph_subdivide.vs)
        for i in range(point_num):
            if self.full_graph_subdivide.degree(i) != 2:
                itsc_indices.add(i)
        self.nms_score_override = np.zeros((point_num, ), dtype=np.float32)
        self.nms_score_override[np.array(list(itsc_indices))] = 2.0  # itsc points will always be kept

        # Points near crossover and intersections are interesting.
        # they will be more frequently sampled
        # 交汇点和横跨点周边的点具备更值得关注，相比于直路上的点，这些点极度影响拓扑
        interesting_indices = set()
        interesting_radius = 32
        # near itsc
        for i in itsc_indices:
            p = self.subdivide_points[i]
            nearby_indices = self.graph_kdtree.query_ball_point(p, interesting_radius)
            interesting_indices.update(nearby_indices)
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(np.array(p), interesting_radius)
            interesting_indices.update(nearby_indices)
        self.sample_weights = np.full((point_num, ), 0.1, dtype=np.float32)
        self.sample_weights[list(interesting_indices)] = 0.9
        
        
        # 清除一下不要的变量，防止内存不够
        del self.full_graph_origin, self.crossover_points, \
            self.graph_kdtree, itsc_indices, \
            interesting_indices, nearby_indices
    
    
        
    def sample_patch(self, patch, rot_index = 0):
        (x0, y0), (x1, y1) = patch
        query_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))    # 找到目标矩形的范围（左上、右下）
        patch_indices_all = set(self.graph_rtee.intersection(query_box))    # 哪些点在这个patch内，注意这里的self.graph_rtee是点的rtree索引
        # 注意这里的图是经过密集化的，不然不足以模拟测试时密集的采样带你的情况
        patch_indices = patch_indices_all - self.exclude_indices

        # Use NMS to downsample, params shall resemble inference time
        patch_indices = np.array(list(patch_indices))
        if len(patch_indices) == 0:
            # print("==== Patch is empty ====")
            # this shall be rare, but if no points in side the patch, return null stuff
            sample_num = self.config.TOPO_SAMPLE_NUM
            max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES
            fake_points = np.array([[0.0, 0.0]], dtype=np.float32)
            fake_sample = ([[0, 0]] * max_nbr_queries, [False] * max_nbr_queries, [False] * max_nbr_queries)
            return fake_points, [fake_sample] * sample_num

        patch_points = self.subdivide_points[patch_indices, :]
        
        # random scores to emulate different random configurations that all share a
        # similar spacing between sampled points
        # raise scores for intersction points so they are always kept
        nms_scores = np.random.uniform(low=0.9, high=1.0, size=patch_indices.shape[0])  # 为每个采样点赋置信度（不简单地赋值为1）
        nms_score_override = self.nms_score_override[patch_indices]
        nms_scores = np.maximum(nms_scores, nms_score_override)     # 挑出两个数组对应位置的最大值组成新数组
        nms_radius = self.config.ROAD_NMS_RADIUS
        
        # kept_indces are into the patch_points array
        nmsed_points, kept_indices = graph_utils.nms_points(patch_points, nms_scores, radius=nms_radius, return_indices=True)
        # now this is into the subdivide graph
        nmsed_indices = patch_indices[kept_indices]
        nmsed_point_num = nmsed_points.shape[0]


        sample_num = self.config.TOPO_SAMPLE_NUM  # has to be greater than 1
        sample_weights = self.sample_weights[nmsed_indices]
        # indices into the nmsed points in the patch
        sample_indices_in_nmsed = np.random.choice(
            np.arange(start=0, stop=nmsed_points.shape[0], dtype=np.int32),
            size=sample_num, replace=True, p=sample_weights / np.sum(sample_weights))   # replace表示是否允许有放回采样
        # indices into the subdivided graph
        sample_indices = nmsed_indices[sample_indices_in_nmsed]
        
        radius = self.config.NEIGHBOR_RADIUS
        max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES  # has to be greater than 1
        nmsed_kdtree = scipy.spatial.KDTree(nmsed_points)
        sampled_points = self.subdivide_points[sample_indices, :]
        # [n_sample, n_nbr]
        # k+1 because the nearest one is always self
        knn_d, knn_idx = nmsed_kdtree.query(sampled_points, k=max_nbr_queries + 1, distance_upper_bound=radius) # +1是为了包含自身


        samples = []

        for i in range(sample_num):
            source_node = sample_indices[i] # 这是在NMS_indices中的索引，而这个索引对应全图上的点的某个点
            valid_nbr_indices = knn_idx[i, knn_idx[i, :] < nmsed_point_num] # 这是在NMS_points中的索引，这个索引只对应NMS中的点
            # 挑选一下返回的近邻点，其在nms_points中的索引不能>=nms点的数量
            # 这貌似有点多余，因为kdtree本来就是用nms_points建立的
            valid_nbr_indices = valid_nbr_indices[1:] # the nearest one is self so remove
            target_nodes = [nmsed_indices[ni] for ni in valid_nbr_indices]  

            ### BFS to find immediate neighbors on graph
            reached_nodes = graph_utils.bfs_with_conditions(self.full_graph_subdivide, source_node, set(target_nodes), radius // self.subdivide_resolution)
            shall_connect = [t in reached_nodes for t in target_nodes]
            ###

            pairs = []
            valid = []
            source_nmsed_idx = sample_indices_in_nmsed[i]
            for target_nmsed_idx in valid_nbr_indices:
                pairs.append((source_nmsed_idx, target_nmsed_idx))
                valid.append(True)

            # zero-pad
            for i in range(len(pairs), max_nbr_queries):
                pairs.append((source_nmsed_idx, source_nmsed_idx))
                shall_connect.append(False)
                valid.append(False)

            samples.append((pairs, shall_connect, valid))

        # Transform points
        # [N, 2]
        nmsed_points -= np.array([x0, y0])[np.newaxis, :]
        # homo for rot
        # [N, 3]
        nmsed_points = np.concatenate([nmsed_points, np.ones((nmsed_point_num, 1), dtype=nmsed_points.dtype)], axis=1)
        trans = np.array([      # 用来平移，把坐标原点移到patch中间
            [1, 0, -0.5 * self.config.PATCH_SIZE],
            [0, 1, -0.5 * self.config.PATCH_SIZE],
            [0, 0, 1],
        ], dtype=np.float32)   
        # ccw 90 deg in img (x, y)
        rot = np.array([   
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        nmsed_points = nmsed_points @ trans.T @ np.linalg.matrix_power(rot.T, rot_index) @ np.linalg.inv(trans.T)   # 乘以逆矩阵是为了把坐标原点又变回patch左上角即(0, 0)
        # matrix_power(rot.T, rot_index)表示应用几次rot矩阵进行变换，这里旋转rot_index次
        nmsed_points = nmsed_points[:, :2]  # 去掉为了齐次而加的1那一列
            
        # Add noise
        noise_scale = 1.0  # pixels
        nmsed_points += np.random.normal(0.0, noise_scale, size=nmsed_points.shape)
    

        return nmsed_points, samples
    

def test_graph_label_generator():
    if not os.path.exists('debug'):
        os.mkdir('debug')

    dataset = 'cityscale'
    if dataset == 'cityscale':
        rgb_path = './cityscale/20cities/region_166_sat.png'
        # Load GT Graph
        gt_graph = pickle.load(open(f"./cityscale/20cities/region_166_refine_gt_graph.p",'rb'))
        coord_transform = lambda v : v[:, ::-1]
    elif dataset == 'spacenet':
        rgb_path = 'spacenet/RGB_1.0_meter/AOI_2_Vegas_210__rgb.png'
        # Load GT Graph
        gt_graph = pickle.load(open(f"spacenet/RGB_1.0_meter/AOI_2_Vegas_210__gt_graph.p",'rb'))
        # gt_graph = pickle.load(open(f"spacenet/RGB_1.0_meter/AOI_4_Shanghai_1061__gt_graph_dense_spacenet.p",'rb'))
        
        coord_transform = lambda v : np.stack([v[:, 1], 400 - v[:, 0]], axis=1)
        # coord_transform = lambda v : v[:, ::-1]
    rgb = read_rgb_img(rgb_path)
    config = addict.Dict()
    config.PATCH_SIZE = 256
    config.ROAD_NMS_RADIUS = 16
    config.TOPO_SAMPLE_NUM = 4
    config.NEIGHBOR_RADIUS = 64
    config.MAX_NEIGHBOR_QUERIES = 16
    gen = GraphLabelGenerator(config, gt_graph, coord_transform)
    patch = ((x0, y0), (x1, y1)) = ((64, 64), (64+config.PATCH_SIZE, 64+config.PATCH_SIZE))
    test_num = 64
    for i in range(test_num):
        rot_index = np.random.randint(0, 4)
        points, samples = gen.sample_patch(patch, rot_index=rot_index)
        rgb_patch = rgb[y0:y1, x0:x1, ::-1].copy()
        rgb_patch = np.rot90(rgb_patch, rot_index, (0, 1)).copy()
        for pairs, shall_connect, valid in samples:
            color = tuple(int(c) for c in np.random.randint(0, 256, size=3))

            for (src, tgt), connected, is_valid in zip(pairs, shall_connect, valid):
                if not is_valid:
                    continue
                p0, p1 = points[src], points[tgt]
                cv2.circle(rgb_patch, p0.astype(np.int32), 4, color, -1)
                cv2.circle(rgb_patch, p1.astype(np.int32), 2, color, -1)
                if connected:
                    cv2.line(
                        rgb_patch,
                        (int(p0[0]), int(p0[1])),
                        (int(p1[0]), int(p1[1])),
                        (255, 255, 255),
                        1,
                    )
        cv2.imwrite(f'debug/viz_{i}.png', rgb_patch)

        
def graph_collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        if key == 'graph_points':
            tensors = [item[key] for item in batch]
            max_point_num = max([x.shape[0] for x in tensors])
            padded = []
            for x in tensors:
                pad_num = max_point_num - x.shape[0]
                padded_x = torch.concat([x, torch.zeros(pad_num, 2)], dim=0)
                padded.append(padded_x)
            collated[key] = torch.stack(padded, dim=0)
        else:
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
    del batch
    return collated



class SatMapDataset(Dataset):
    def __init__(self, config, is_train, dev_run=False):
        self.config = config
        
        assert self.config.DATASET in {'cityscale', 'spacenet'}
        if self.config.DATASET == 'cityscale':
            self.IMAGE_SIZE = 2048
            # TODO: SAMPLE_MARGIN here is for training, the one in config is for inference
            self.SAMPLE_MARGIN = 64

            self.rgb_pattern = './cityscale/20cities/region_{}_sat.png'
            self.keypoint_mask_pattern = './cityscale/processed/keypoint_mask_{}.png'
            self.road_mask_pattern = './cityscale/processed/road_mask_{}.png'
            self.gt_graph_pattern = './cityscale/20cities/region_{}_refine_gt_graph.p'
            self.GTE_pattern = './cityscale/GTE/region_{}_GTE.npz'
            
            train, val, test = cityscale_data_partition()

            # coord-transform = (r, c) -> (x, y)
            # takes [N, 2] points
            self.coord_transform = lambda v : v[:, ::-1]

        self.is_train = is_train

        train_split = train + val
        test_split = test

        tile_indices = train_split if self.is_train else test_split
        self.tile_indices_full = tile_indices
        
        
        # Stores all imgs in memory.
        self.rgbs, self.keypoint_masks, self.road_masks = [], [], []
        self.GTEs = []
        self.max_degree = 6
        # For graph label generation.
        self.graph_label_generators = []


        ##### FAST DEBUG
        self.set_tileindex_range(intact=True)
        if self.is_train and self.config.DATA_SPLIT:
            self.set_tileindex_range(intact=False, start=0, end=len(self.tile_indices_full)//2)
            
        ##### FAST DEBUG
        if dev_run:
            self.tile_indices = self.tile_indices_full[:4]
        self.load_data()
            
        
        self.sample_min = self.SAMPLE_MARGIN
        self.sample_max = self.IMAGE_SIZE - (self.config.PATCH_SIZE + self.SAMPLE_MARGIN)
        
        self.noise_mask = (np.random.rand(64,64,3)) * 1.0 + 0.5

        if not self.is_train:
            eval_patches_per_edge = math.ceil((self.IMAGE_SIZE - 2 * self.SAMPLE_MARGIN) / self.config.PATCH_SIZE)  # 这句代码使得在首尾抠除margin大小后，eval时只有除不尽时才会有重叠，不够的部分会靠均匀地重叠来达成
            self.eval_patches = []
            for i in range(len(self.tile_indices)):
                self.eval_patches += get_patch_info_one_img(
                    i, self.IMAGE_SIZE, self.SAMPLE_MARGIN, self.config.PATCH_SIZE, eval_patches_per_edge
                )   


    def load_data(self):
        print(" ======== Loading Data ======== ")
        for tile_idx in tqdm(self.tile_indices):
            # print(f'loading tile {tile_idx}')
            rgb_path = self.rgb_pattern.format(tile_idx)
            GTE_path = self.GTE_pattern.format(tile_idx)
            gt_graph_adj = pickle.load(open(self.gt_graph_pattern.format(tile_idx),'rb'))    # 为新增的可学习的topo-decoder构造训练数据
            if len(gt_graph_adj) == 0:
                print(f" ======== skip empty tile {tile_idx} ========")
                continue
            
            graph_label_generator = GraphLabelGenerator(config=self.config, full_graph=gt_graph_adj, coord_transform=self.coord_transform)
            self.rgbs.append(read_rgb_img(rgb_path))  
            self.GTEs.append(np.load(GTE_path)['GTE'])
            # self.rgbs.append(np.zeros((2048, 2048, 3), dtype=np.uint8))  
            # self.GTEs.append(np.zeros((2048, 2048, 19), dtype=np.float32))
            self.graph_label_generators.append(graph_label_generator)
            
            del graph_label_generator


    def set_tileindex_range(self, intact=True, start=0, end=0):
        if intact:
            self.tile_indices = self.tile_indices_full
        else:
            assert start != end
            self.tile_indices = self.tile_indices_full[start:end]
        
    
        
    def __len__(self):
        if self.is_train:
            # Pixel seen in one epoch ~ 17 x total pixels in training set
            if self.config.DATASET == 'cityscale':
                if self.config.dev_run:
                    return 2048
                return max(1, int(self.IMAGE_SIZE / self.config.PATCH_SIZE)) ** 2 * 2500 
                
            elif self.config.DATASET == 'spacenet':
                return 84667
        else:
            return len(self.eval_patches)

    def __getitem__(self, idx):
        # Sample a patch.ADAD
        if self.is_train:
            img_idx = np.random.randint(low=0, high=len(self.rgbs))
            begin_x = np.random.randint(low=self.sample_min, high=self.sample_max+1)
            begin_y = np.random.randint(low=self.sample_min, high=self.sample_max+1)
            end_x, end_y = begin_x + self.config.PATCH_SIZE, begin_y + self.config.PATCH_SIZE
        else:
            # Returns eval patch
            img_idx, (begin_x, begin_y), (end_x, end_y) = self.eval_patches[idx]
        
        # Crop patch imgs and masks
        rgb_patch = self.rgbs[img_idx][begin_y:end_y, begin_x:end_x, :]
        GTE_patch = self.GTEs[img_idx][begin_y:end_y, begin_x:end_x, :]    # GTE也是按照rc编码的，但是samRoad使用的graph是xy的

        # Augmentation
        rot_index = 0
        if self.is_train:
            rot_index = np.random.randint(0, 4)
            # CCWs
            rgb_patch = np.rot90(rgb_patch, rot_index, [0,1]).copy()
            GTE_patch = np.rot90(GTE_patch, rot_index, [0, 1]).copy()
            
            # 图片加点噪声
            if np.random.randint(0, 100) < 50:
                for _ in range(np.random.randint(1,5)):
                    xx = np.random.randint(0,self.config.PATCH_SIZE-64-1)
                    yy = np.random.randint(0,self.config.PATCH_SIZE-64-1)

                    rgb_patch[xx:xx+64,yy:yy+64,:] = np.multiply(rgb_patch[xx:xx+64,yy:yy+64,:] + 0.5, self.noise_mask) - 0.5
                
				# add more noise 
                for _ in range(np.random.randint(1,3)):
                    xx = np.random.randint(0,self.config.PATCH_SIZE-64-1)
                    yy = np.random.randint(0,self.config.PATCH_SIZE-64-1)

                    rgb_patch[xx:xx+64,yy:yy+64,:] =  (self.noise_mask - 1.0) 
            
            
            # dx, dy的变换
            # rot_mat = np.array([    # 只针对90度的旋转，如果是任意角度还需要改旋转矩阵
            #             [math.cos(math.pi/2), math.sin(math.pi/2)],
            #             [-math.sin(math.pi/2), math.cos(math.pi/2)]s
            # ], dtype=np.float32)
            
            rot_mat = np.array([    # 只针对90度的旋转，如果是任意角度还需要改旋转矩阵
                        [0, 1],
                        [-1, 0]
            ], dtype=np.float32)

            # 只对所有不为0的进行变换，其余的全是0，没必要再计算了
            for r, c in np.column_stack(np.where(GTE_patch[:,:, 0]>0.98)):
                delta_coords = []
                for j in range(self.max_degree):
                    delta_coords.append([GTE_patch[r, c, 1+3*j+1], GTE_patch[r, c, 1+3*j+2]])   # dx, dy
                delta_coords_to_rot = np.column_stack(delta_coords)
                roted_coords = np.linalg.matrix_power(rot_mat, rot_index)@delta_coords_to_rot
                
                for j in range(self.max_degree):    # stick back
                    GTE_patch[r, c, 1+3*j+1], GTE_patch[r, c, 1+3*j+2] = roted_coords[:, j]
                
                    
        patch = ((begin_x, begin_y), (end_x, end_y))
        graph_points, topo_samples = self.graph_label_generators[img_idx].sample_patch(patch, rot_index)
        pairs, connected, valid = zip(*topo_samples)
        

        return {
            # GTE
            'rgb': torch.tensor(rgb_patch, dtype=torch.float32),
            'GTE': torch.tensor(GTE_patch, dtype=torch.float32),
            # topo
            'graph_points': torch.tensor(graph_points, dtype=torch.float32),
            'pairs': torch.tensor(pairs, dtype=torch.int32),
            'connected': torch.tensor(connected, dtype=torch.bool),
            'valid': torch.tensor(valid, dtype=torch.bool),
        }

if __name__ == '__main__':
    test_graph_label_generator()
    # train, val, test = cityscale_data_partition()
    # print(f'cityscale train {len(train)} val {len(val)} test {len(test)}')
    # train, val, test = spacenet_data_partition()
    # print(f'spacenet train {len(train)} val {len(val)} test {len(test)}')
