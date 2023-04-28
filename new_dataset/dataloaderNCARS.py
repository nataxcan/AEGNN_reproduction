import functools
import glob
import logging
import numpy as np
import os
import torch
import abc
import argparse

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import FixedPoints
from torch_geometric.transforms import Cartesian
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Union
from torch.utils.data.dataset import Dataset, T_co
# import pytorch_lightning as pl
import torch_geometric
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset
from multiprocess import Pool

# from aegnn.utils.bounding_box import crop_to_frame
# from aegnn.utils.multiprocessing import TaskManager
# from aegnn.datasets.base.event_dm import EventDataModule
# from aegnn.datasets.utils.normalization import normalize_time

from typing import Callable, List, Optional, Tuple

class NCarsBest(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, mode='train'):
        super(NCarsBest, self).__init__(root, transform, pre_transform)
        
        # TODO: Initialize your dataset here
        # self.data_dir = './data/storage/'
        self.data_dir = './data/storage/N-Cars_parsed/'
        # test_folder = 'ncars/n-cars_test/'
        # training_folder = 'ncars/n-cars_train/'
        test_folder = 'test/'
        training_folder = 'train/'
        validation_folder = 'val/'
        # self.classes = {}
        # self.class_indexes = {}
        # for index, folder in enumerate(os.listdir(self.data_dir+test_folder)):
        #     # first get all classes in a list
        #     self.classes[str(folder)] = index
        #     self.class_indexes[index] = str(folder)
        print("loading classes...")
        # print(self.classes)
        self.train_data = list(map(lambda x: os.path.join(self.data_dir + training_folder, x),os.listdir(self.data_dir + training_folder))) + list(map(lambda x: os.path.join(self.data_dir + validation_folder, x),os.listdir(self.data_dir + validation_folder)))
        self.test_data = list(map(lambda x: os.path.join(self.data_dir + test_folder, x),os.listdir(self.data_dir + test_folder)))
        print(self.train_data)
        print("Train data: ", len(self.train_data))
        print("Test data: ", len(self.test_data))
        # with Pool(8) as p:
        #     for folder in tqdm(os.listdir(self.data_dir) + , total=len(self.data_dir)):
        #         li1 = [self.data_dir + test_folder + folder + "/" + fn for fn in os.listdir(self.data_dir + test_folder + folder)]
        #         li2 = [self.data_dir + training_folder + folder + "/" + fn for fn in os.listdir(self.data_dir + training_folder + folder)]
        #         # res1 = p.map(self.load, li1)
        #         # res2 = p.map(self.load, li2)
        #         # res3 = p.map(self.load, li3)
        #         self.test_data += li1
        #         self.train_data += li2
        # print(len(self.train_data))
        # self.test_data, self.train_data, self.val_data = [], [], []
        # for folder in tqdm(os.listdir(self.data_dir+test_folder), total=len(self.classes)):
        #     self.test_data += [self.load(self.data_dir + test_folder + folder + "/" + fn) for fn in os.listdir(self.data_dir + test_folder + folder)]
        #     self.train_data += [self.load(self.data_dir + training_folder + folder + "/" + fn) for fn in os.listdir(self.data_dir + training_folder + folder)]
        #     self.val_data += [self.load(self.data_dir + validation_folder + folder + "/" + fn) for fn in os.listdir(self.data_dir + validation_folder + folder)]
        self.data = []
        if mode == 'train':
            self.data = self.train_data
        if mode == 'test':
            self.data = self.test_data

    # @functools.lru_cache(maxsize=100)
    # def map_label(self, label: str) -> int:
    #     return self.classes[label]

    # def get_label(self, label: int) -> str:
    #     return self.class_indexes[label]


    def load(self, raw_file: str) -> Data:
        events_file = os.path.join(raw_file, "events.txt")
        events = torch.from_numpy(np.loadtxt(events_file)).float()
        x, pos = events[:, -1:], events[:, :3]
        y = self.read_label(raw_file)
        # print(x[0:2], pos[0:2], y)
        data = Data(x=x, y = y, pos=pos)
        data = self.process_data(data)
        data = Cartesian(norm=True, cat=False, max_value=10.0)(data)
        return data
    
    def read_label(self, raw_file: str) -> Optional[Union[str, List[str]]]:
        label_file = os.path.join(raw_file, "is_car.txt")
        with open(label_file, "r") as f:
            label_txt = f.read().replace(" ", "").replace("\n", "")
        return 1 if label_txt == "1" else 0

    # def load(self, raw_file: str) -> Data:
    #     f = open(raw_file, 'rb')
    #     endOfHeader = False
    #     numCommentLine = 0

    #     raw_data = []
    #     while not endOfHeader and not numCommentLine >=3:
    #         tline = f.readline().decode('utf-8')
    #         if not tline.startswith('%'):
    #             endOfHeader = True
    #         else:
    #             numCommentLine += 1
    #         # Seek back to the beginning of the line
    #     raw_data = f.read()
    #     # print(raw_data[])
    #     raw_data = np.frombuffer(raw_data, dtype=np.uint32, offset=2)
    #     binarr = np.unpackbits(raw_data.view(np.uint8)).reshape(-1,32)
    #     reversedbinarr = binarr[:,::-1]
    #     raw_data = np.packbits(reversedbinarr).view(np.uint32)
    #     # raw_data = np.uint32(raw_data)
    #     print(raw_data[0:6])
    #     all_y = (raw_data[1::2] >> 1) & 0x3FFF
    #     all_x = (raw_data[1::2] >> 15) & 0x3FFF
    #     all_p = (raw_data[1::2]) & 1  # bit 7
    #     all_ts = raw_data[0::2]

    #     print(all_y[0:6], all_x[0:6], all_ts[0:6])
    #     all_ts = all_ts / 1e6  # µs -> s
    #     all_p = all_p.astype(np.float64)
    #     all_p[all_p == 0] = -1
    #     s = np.min([len(all_x), len(all_y), len(all_ts), len(all_p)])
    #     all_x, all_y, all_ts, all_p = all_x[:s], all_y[:s], all_ts[:s], all_p[:s]
    #     events = np.column_stack((all_x, all_y, all_ts, all_p))
    #     # except Exception as e:
    #     #     print("tada:", all_x.shape, all_y.shape, all_ts.shape, all_p.shape)
    #     events = torch.from_numpy(events).float()

    #     x, pos = events[:, -1:], events[:, :3]   # x = polarity, pos = spatio-temporal position
    #     # print("X", x)
    #     # print("POS", pos)
    #     # print(x.type(), pos.type())
    #     data = Data(
    #             x=x, # polarity
    #             pos=pos, # in order: x_pos, y_pos, timestamp
    #             y=self.read_label(raw_file), # int representing output head index that should be max
    #     )
        
    #     # data.to('cuda')
    #     data = self.process_data(data)
    #     data = torch_geometric.transforms.Cartesian(norm=True, cat=False, max_value=10.0)(data)
    #     # data.to('cpu')

    #     return data
    
    # def read_label(self, raw_file: str) -> Optional[Union[str, List[str]]]:
    #     return self.map_label(raw_file.split("/")[-2])


    @property
    def raw_file_names(self):
        # TODO: Return a list of raw file names
        return []
    
    @property
    def processed_file_names(self):
        # TODO: Return a list of processed file names
        return []
    
    def download(self):
        # TODO: Download the dataset if needed
        pass
    
    def process(self):
        # TODO: Process the raw data into the processed data
        pass
    
    def len(self):
        return len(self.data)

    
    def get(self, idx):
        assert idx < len(self.data)
        return self.load(self.data[idx])

    def process_data(self, data: Data) -> Data:
        # params = self.hparams.preprocessing
        params = {}
        params['sampling'] = True
        params['r'] = 3
        params['d_max'] = 32

        # print("before shiz", data.num_nodes)
        # Cut-off window of highest increase of events.
        # window_us = 1 * 1000
        # t = data.pos[data.num_nodes // 2, 2]
        # index1 = torch.clamp(torch.searchsorted(data.pos[:, 2].contiguous(), t) - 1, 0, data.num_nodes - 1)
        # index0 = torch.clamp(torch.searchsorted(data.pos[:, 2].contiguous(), t-window_us) - 1, 0, data.num_nodes - 1)
        # numnodes = data.num_nodes
        # not_tensor = 0
        # not_size_equal = 0
        # size_one = 0
        # for key, item in data:
        #     # print(key)
        #     if not torch.is_tensor(item):
        #         not_tensor += 1
        #     elif item.size(0) == numnodes:
        #         not_size_equal += 1
        #     elif not item.size(0) != 1:
        #         size_one += 1
        #     if torch.is_tensor(item) and item.size(0) == numnodes and item.size(0) != 1:
        #         data[key] = item[index0:index1, :]
        # print("after shizz", data.num_nodes)
        # print(not_tensor)
        # print(not_size_equal)
        # print(size_one)
        # set number of sub-samples to 1/coeff times the number of samples
        # SAMPLING_COEFFICIENT = 10
        params['n_samples'] = 1000

        # Coarsen graph by uniformly sampling n points from the event point cloud.
        data = self.sub_sampling(data, n_samples=params["n_samples"], sub_sample=params["sampling"])
        # print("after sample", data.num_nodes)
        # Re-weight temporal vs. spatial dimensions to account for different resolutions.
        data.pos[:, 2] = self.normalize_time(data.pos[:, 2])
        # Radius graph generation.
        data.edge_index = radius_graph(data.pos, r=params["r"], max_num_neighbors=params["d_max"])
        return data
    
    def sub_sampling(self, data: Data, n_samples: int, sub_sample: bool) -> Data:
        if sub_sample:
            sampler = FixedPoints(num=n_samples, allow_duplicates=False, replace=False)
            sampled = sampler(data)
            return sampled
        else:
            sample_idx = np.arange(n_samples)
            for key, item in data:
                if torch.is_tensor(item) and item.size(0) != 1:
                    data[key] = item[sample_idx]
            return data
    
    def normalize_time(self, ts: torch.Tensor, beta: float = 0.5e-5) -> torch.Tensor:
        """Normalizes the temporal component of the event pos by using beta re-scaling

        :param ts: time-stamps to normalize in microseconds [N].
        :param beta: re-scaling factor.
        """
        return (ts - torch.min(ts)) * beta