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


class NCaltech101Best(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, num_samples=50, mode='train'):
        super(NCaltech101Best, self).__init__(root, transform, pre_transform)

        self.num_samples = num_samples
        
        # TODO: Initialize your dataset here
        self.data_dir = './data/storage/'
        test_folder = 'ncaltech101/test/'
        training_folder = 'ncaltech101/training/'
        validation_folder = 'ncaltech101/validation/'

        self.classes = {}
        self.class_indexes = {}
        folderlist = os.listdir(self.data_dir+test_folder)
        folderlist.sort()
        for index, folder in enumerate(folderlist):
            # first get all classes in a list
            self.classes[str(folder)] = index
            self.class_indexes[index] = str(folder)
        print("loading classes...")

        self.test_data, self.train_data, self.val_data = [], [], []

        with Pool(8) as p:
            for folder in tqdm(os.listdir(self.data_dir+test_folder), total=len(self.classes)):
                li1 = [self.data_dir + test_folder + folder + "/" + fn for fn in os.listdir(self.data_dir + test_folder + folder)]
                li2 = [self.data_dir + training_folder + folder + "/" + fn for fn in os.listdir(self.data_dir + training_folder + folder)]
                li3 = [self.data_dir + validation_folder + folder + "/" + fn for fn in os.listdir(self.data_dir + validation_folder + folder)]
                # res1 = p.map(self.load, li1)
                # res2 = p.map(self.load, li2)
                # res3 = p.map(self.load, li3)
                self.test_data += li1
                self.train_data += li2
                self.val_data += li3
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
        if mode == 'eval':
            self.data = self.val_data
    
    @functools.lru_cache(maxsize=100)
    def map_label(self, label: str) -> int:
        return self.classes[label]

    def get_label(self, label: int) -> str:
        return self.class_indexes[label]

    def load(self, raw_file: str) -> Data:
        f = open(raw_file, 'rb')
        raw_data = np.fromfile(f, dtype=np.uint8)
        f.close()

        raw_data = np.uint32(raw_data)
        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7  # bit 7

        a, b, c = ((raw_data[2::5] & 127) << 16), (raw_data[3::5] << 8), (raw_data[4::5])
        s = np.min([len(a), len(b), len(c)])
        a, b, c = a[:s], b[:s], c[:s]
        all_ts = a | b | c

        all_ts = all_ts / 1e6  # µs -> s
        all_p = all_p.astype(np.float64)
        all_p[all_p == 0] = -1
        s = np.min([len(all_x), len(all_y), len(all_ts), len(all_p)])
        all_x, all_y, all_ts, all_p = all_x[:s], all_y[:s], all_ts[:s], all_p[:s]
        events = np.column_stack((all_x, all_y, all_ts, all_p))
        # except Exception as e:
        #     print("tada:", all_x.shape, all_y.shape, all_ts.shape, all_p.shape)
        events = torch.from_numpy(events).float()

        x, pos = events[:, -1:], events[:, :3]   # x = polarity, pos = spatio-temporal position
        # print("X", x)
        # print("POS", pos)
        data = Data(
                x=x, # polarity
                pos=pos, # in order: x_pos, y_pos, timestamp
                y=self.read_label(raw_file), # int representing output head index that should be max
        )
        
        # data.to('cuda')
        data = self.process_data(data)
        data = torch_geometric.transforms.Cartesian(norm=True, cat=False, max_value=10.0)(data)
        # data.to('cpu')

        return data
    
    def read_label(self, raw_file: str) -> Optional[Union[str, List[str]]]:
        return self.map_label(raw_file.split("/")[-2])


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
        params['r'] = 8
        params['d_max'] = 6


        # Cut-off window of highest increase of events.
        window_us = 50 * 1000
        t = data.pos[data.num_nodes // 2, 2]
        index1 = torch.clamp(torch.searchsorted(data.pos[:, 2].contiguous(), t) - 1, 0, data.num_nodes - 1)
        index0 = torch.clamp(torch.searchsorted(data.pos[:, 2].contiguous(), t-window_us) - 1, 0, data.num_nodes - 1)
        numnodes = data.num_nodes
        for key, item in data:
            if torch.is_tensor(item) and item.size(0) == numnodes and item.size(0) != 1:
                data[key] = item[index0:index1, :]

        # set number of sub-samples to 1/coeff times the number of samples
        SAMPLING_COEFFICIENT = 10
        params['n_samples'] = self.num_samples

        # Coarsen graph by uniformly sampling n points from the event point cloud.
        data = self.sub_sampling(data, n_samples=params["n_samples"], sub_sample=params["sampling"])
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






