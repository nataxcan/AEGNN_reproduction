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
from torch_geometric.data import InMemoryDataset

# from aegnn.utils.bounding_box import crop_to_frame
# from aegnn.utils.multiprocessing import TaskManager
# from aegnn.datasets.base.event_dm import EventDataModule
# from aegnn.datasets.utils.normalization import normalize_time

from typing import Callable, List, Optional, Tuple


class NCaltech101Best(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, mode='train'):
        super(NCaltech101Best, self).__init__(root, transform, pre_transform)
        
        # TODO: Initialize your dataset here
        self.data_dir = './data/storage/'
        test_folder = 'ncaltech101/test/'
        training_folder = 'ncaltech101/training/'
        validation_folder = 'ncaltech101/validation/'

        self.classes = {}
        self.class_indexes = {}
        for index, folder in enumerate(os.listdir(self.data_dir+test_folder)):
            # first get all classes in a list
            self.classes[str(folder)] = index
            self.class_indexes[index] = str(folder)
        print("loading classes...")

        self.test_data, self.train_data, self.val_data = [], [], []
        for folder in tqdm(os.listdir(self.data_dir+test_folder), total=len(self.classes)):
            self.test_data += [self.load(self.data_dir + test_folder + folder + "/" + fn) for fn in os.listdir(self.data_dir + test_folder + folder)]
            self.train_data += [self.load(self.data_dir + training_folder + folder + "/" + fn) for fn in os.listdir(self.data_dir + training_folder + folder)]
            self.val_data += [self.load(self.data_dir + validation_folder + folder + "/" + fn) for fn in os.listdir(self.data_dir + validation_folder + folder)]
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
        events = torch.from_numpy(events).float().cuda()

        x, pos = events[:, -1:], events[:, :3]   # x = polarity, pos = spatio-temporal position
        # print("X", x)
        # print("POS", pos)
        data = Data(
                x=x, # polarity
                pos=pos, # in order: x_pos, y_pos, timestamp
                y=self.read_label(raw_file), # int representing output head index that should be max
        )
        
        data = self.process_data(data)
        data.edge_attr = torch_geometric.transforms.Cartesian(cat=False, max_value=10.0)(data).edge_attr

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
        return self.data[idx]

    def process_data(self, data: Data) -> Data:
        # params = self.hparams.preprocessing
        params = {}
        params['n_samples'] = 100
        params['sampling'] = 10
        params['r'] = 8
        params['d_max'] = 6

        # Cut-off window of highest increase of events.
        window_us = 50 * 1000
        t = data.pos[data.num_nodes // 2, 2]
        index1 = torch.clamp(torch.searchsorted(data.pos[:, 2].contiguous(), t) - 1, 0, data.num_nodes - 1)
        index0 = torch.clamp(torch.searchsorted(data.pos[:, 2].contiguous(), t-window_us) - 1, 0, data.num_nodes - 1)
        for key, item in data:
            if torch.is_tensor(item) and item.size(0) == data.num_nodes and item.size(0) != 1:
                data[key] = item[index0:index1, :]

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
            return sampler(data)
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






# class NCalTech101Better(pl.LightningDataModule):


#     def __init__(self, data_dir: str = "./data/storage/"):
#         super().__init__()
        
#         self.data_dir = data_dir
#         self.transform = lambda x: x

#         # self.dims is returned when you call dm.size()
#         # Setting default dims here because we know them.
#         # Could optionally be assigned dynamically in dm.setup()
#         self.dims = (2, 240, 180)
        

#     def prepare_data(self):
#         # download
#         # self.load_whole_dataset()
#         pass

#     def setup(self, stage: Optional[str] = None):
#         self.load_whole_dataset()
        

#     def train_dataloader(self):
#         return DataLoader(self.train_data, batch_size=16)

#     def val_dataloader(self):
#         return DataLoader(self.val_data, batch_size=16)

#     def test_dataloader(self):
#         return DataLoader(self.test_data, batch_size=16)
    
#     def load_whole_dataset(self) -> List[Data]:
#         test_folder = 'ncaltech101/test/'
#         training_folder = 'ncaltech101/training/'
#         validation_folder = 'ncaltech101/validation/'
#         self.test_data, self.train_data, self.val_data = [], [], []
#         for folder in os.listdir(self.data_dir+test_folder):
#             self.test_data += [self.load(self.data_dir + test_folder + folder + "/" + fn) for fn in os.listdir(self.data_dir + test_folder + folder)]
#             self.train_data += [self.load(self.data_dir + training_folder + folder + "/" + fn) for fn in os.listdir(self.data_dir + training_folder + folder)]
#             self.val_data += [self.load(self.data_dir + validation_folder + folder + "/" + fn) for fn in os.listdir(self.data_dir + validation_folder + folder)]

    
#     def load(self, raw_file: str) -> Data:
#         f = open(raw_file, 'rb')
#         raw_data = np.fromfile(f, dtype=np.uint8)
#         f.close()

#         raw_data = np.uint32(raw_data)
#         all_y = raw_data[1::5]
#         all_x = raw_data[0::5]
#         all_p = (raw_data[2::5] & 128) >> 7  # bit 7

#         a, b, c = ((raw_data[2::5] & 127) << 16), (raw_data[3::5] << 8), (raw_data[4::5])
#         s = np.min([len(a), len(b), len(c)])
#         a, b, c = a[:s], b[:s], c[:s]
#         all_ts = a | b | c

#         all_ts = all_ts / 1e6  # µs -> s
#         all_p = all_p.astype(np.float64)
#         all_p[all_p == 0] = -1
#         s = np.min([len(all_x), len(all_y), len(all_ts), len(all_p)])
#         all_x, all_y, all_ts, all_p = all_x[:s], all_y[:s], all_ts[:s], all_p[:s]
#         events = np.column_stack((all_x, all_y, all_ts, all_p))
#         # except Exception as e:
#         #     print("tada:", all_x.shape, all_y.shape, all_ts.shape, all_p.shape)
#         events = torch.from_numpy(events).float().cuda()

#         x, pos = events[:, -1:], events[:, :3]   # x = polarity, pos = spatio-temporal position
#         return Data(x=x, pos=pos, y=self.read_label(raw_file))
    
#     def read_label(self, raw_file: str) -> Optional[Union[str, List[str]]]:
#         return raw_file.split("/")[-2]


# class NCaltech101(EventDataModule):

#     def __init__(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 8,
#                  pin_memory: bool = False, transform: Optional[Callable[[Data], Data]] = None):
#         super(NCaltech101, self).__init__(img_shape=(240, 180), batch_size=batch_size, shuffle=shuffle,
#                                           num_workers=num_workers, pin_memory=pin_memory, transform=transform)
#         pre_processing_params = {"r": 5.0, "d_max": 32, "n_samples": 25000, "sampling": True}
#         self.save_hyperparameters({"preprocessing": pre_processing_params})

#     def read_annotations(self, raw_file: str) -> Optional[np.ndarray]:
#         annotations_dir = os.path.join(os.environ["AEGNN_DATA_DIR"], "ncaltech101", "annotations")
#         raw_file_name = os.path.basename(raw_file).replace("image", "annotation")
#         raw_dir_name = os.path.basename(os.path.dirname(raw_file))
#         annotation_file = os.path.join(os.path.join(annotations_dir, raw_dir_name, raw_file_name))

#         f = open(annotation_file)
#         annotations = np.fromfile(f, dtype=np.int16)
#         annotations = np.array(annotations[2:10])
#         f.close()

#         label = self.read_label(raw_file)
#         class_id = self.map_label(label)
#         if class_id is None:
#             return None

#         # Create bounding box from corner, shape and label variables. NCaltech101 bounding boxes
#         # often start outside of the frame (negative corner coordinates). However, the shape turns
#         # out to be the shape of the bbox starting at the image's frame.
#         bbox = np.array([
#             annotations[0], annotations[1],  # upper-left corner
#             annotations[2] - annotations[0],  # width
#             annotations[5] - annotations[1],  # height
#             class_id
#         ])
#         bbox[:2] = np.maximum(bbox[:2], 0)
#         return bbox.reshape((1, 1, -1))

#     @staticmethod
#     def read_label(raw_file: str) -> Optional[Union[str, List[str]]]:
#         return raw_file.split("/")[-2]

#     @staticmethod
#     def load(raw_file: str) -> Data:
#         f = open(raw_file, 'rb')
#         raw_data = np.fromfile(f, dtype=np.uint8)
#         f.close()

#         raw_data = np.uint32(raw_data)
#         all_y = raw_data[1::5]
#         all_x = raw_data[0::5]
#         all_p = (raw_data[2::5] & 128) >> 7  # bit 7
#         all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
#         all_ts = all_ts / 1e6  # µs -> s
#         all_p = all_p.astype(np.float64)
#         all_p[all_p == 0] = -1
#         events = np.column_stack((all_x, all_y, all_ts, all_p))
#         events = torch.from_numpy(events).float().cuda()

#         x, pos = events[:, -1:], events[:, :3]   # x = polarity, pos = spatio-temporal position
#         return Data(x=x, pos=pos)

#     @functools.lru_cache(maxsize=100)
#     def map_label(self, label: str) -> int:
#         label_dict = {lbl: i for i, lbl in enumerate(self.classes)}
#         return label_dict.get(label, None)

#     def _load_processed_file(self, f_path: str) -> Data:
#         return torch.load(f_path)

#     #########################################################################################################
#     # Processing ############################################################################################
#     #########################################################################################################
#     def _prepare_dataset(self, mode: str):
#         processed_dir = os.path.join(self.root, "processed")
#         raw_files = self.raw_files(mode)
#         class_dict = {class_id: i for i, class_id in enumerate(self.classes)}
#         kwargs = dict(load_func=self.load, class_dict=class_dict, pre_transform=self.pre_transform,
#                       read_label=self.read_label, read_annotations=self.read_annotations)
#         logging.debug(f"Found {len(raw_files)} raw files in dataset (mode = {mode})")

#         task_manager = TaskManager(self.num_workers, queue_size=self.num_workers)
#         processed_files = []
#         for rf in tqdm(raw_files):
#             processed_file = rf.replace(self.root, processed_dir)
#             processed_files.append(processed_file)

#             if os.path.exists(processed_file):
#                 continue
#             task_manager.queue(self.processing, rf=rf, pf=processed_file, **kwargs)
#         task_manager.join()

#     @staticmethod
#     def processing(rf: str, pf: str, load_func: Callable[[str], Data],
#                    class_dict: Dict[str, int], read_label: Callable[[str], str],
#                    read_annotations: Callable[[str], np.ndarray], pre_transform: Callable = None):
#         rf_wo_ext, _ = os.path.splitext(rf)

#         # Load data from raw file. If the according loaders are available, add annotation, label and class id.
#         device = "cpu"  # torch.device(torch.cuda.current_device())
#         data_obj = load_func(rf).to(device)
#         data_obj.file_id = os.path.basename(rf)
#         if (label := read_label(rf)) is not None:
#             data_obj.label = label if isinstance(label, list) else [label]
#             data_obj.y = torch.tensor([class_dict[label] for label in data_obj.label])
#         if (bbox := read_annotations(rf)) is not None:
#             data_obj.bbox = torch.tensor(bbox, device=device).long()

#         # Apply the pre-transform on the graph, to afterwards store it as .pt-file.
#         assert data_obj.pos.size(1) == 3, "pos must consist of (x, y, t)"
#         if pre_transform is not None:
#             data_obj = pre_transform(data_obj)

#         # Save the data object as .pt-torch-file. For the sake of a uniform processed
#         # directory format make all output paths flat.
#         os.makedirs(os.path.dirname(pf), exist_ok=True)
#         torch.save(data_obj.to("cpu"), pf)

#     def pre_transform(self, data: Data) -> Data:
#         params = self.hparams.preprocessing

#         # Cut-off window of highest increase of events.
#         window_us = 50 * 1000
#         t = data.pos[data.num_nodes // 2, 2]
#         index1 = torch.clamp(torch.searchsorted(data.pos[:, 2].contiguous(), t) - 1, 0, data.num_nodes - 1)
#         index0 = torch.clamp(torch.searchsorted(data.pos[:, 2].contiguous(), t-window_us) - 1, 0, data.num_nodes - 1)
#         for key, item in data:
#             if torch.is_tensor(item) and item.size(0) == data.num_nodes and item.size(0) != 1:
#                 data[key] = item[index0:index1, :]

#         # Coarsen graph by uniformly sampling n points from the event point cloud.
#         data = self.sub_sampling(data, n_samples=params["n_samples"], sub_sample=params["sampling"])

#         # Re-weight temporal vs. spatial dimensions to account for different resolutions.
#         data.pos[:, 2] = normalize_time(data.pos[:, 2])
#         # Radius graph generation.
#         data.edge_index = radius_graph(data.pos, r=params["r"], max_num_neighbors=params["d_max"])
#         return data

#     @staticmethod
#     def sub_sampling(data: Data, n_samples: int, sub_sample: bool) -> Data:
#         if sub_sample:
#             sampler = FixedPoints(num=n_samples, allow_duplicates=False, replace=False)
#             return sampler(data)
#         else:
#             sample_idx = np.arange(n_samples)
#             for key, item in data:
#                 if torch.is_tensor(item) and item.size(0) != 1:
#                     data[key] = item[sample_idx]
#             return data

#     #########################################################################################################
#     # Files #################################################################################################
#     #########################################################################################################
#     def raw_files(self, mode: str) -> List[str]:
#         return glob.glob(os.path.join(self.root, mode, "*", "*.bin"), recursive=True)

#     def processed_files(self, mode: str) -> List[str]:
#         processed_dir = os.path.join(self.root, "processed")
#         return glob.glob(os.path.join(processed_dir, mode, "*", "*.bin"))

#     @property
#     def classes(self) -> List[str]:
#         ospath = os.path.join(self.root, "raw")
#         print(ospath[1:])
#         return os.listdir(ospath[1:])


# class EventDataModule(pl.LightningDataModule):

#     def __init__(self, img_shape: Tuple[int, int], batch_size: int, shuffle: bool, num_workers: int,
#                  pin_memory: bool, transform: Optional[Callable[[Data], Data]] = None):
#         # super(EventDataModule, self).__init__(dims=img_shape) # THIS CHANGED BY NATHAN

#         self.num_workers = num_workers
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.pin_memory = pin_memory

#         self.train_dataset = None
#         self.val_dataset = None
#         self.transform = transform

#     def prepare_data(self) -> None:
#         logging.info("Preparing datasets for loading")
#         self._prepare_dataset("training")
#         self._prepare_dataset("validation")

#     def setup(self, stage: Optional[str] = None):
#         logging.debug("Load and set up datasets")
#         self.train_dataset = self._load_dataset("training")
#         self.val_dataset = self._load_dataset("validation")
#         if len(self.train_dataset) == 0 or len(self.val_dataset) == 0:
#             raise UserWarning("No data found, check AEGNN_DATA_DIR environment variable!")

#     #########################################################################################################
#     # Data Loaders ##########################################################################################
#     #########################################################################################################
#     def train_dataloader(self) -> torch.utils.data.DataLoader:
#         return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
#                                            num_workers=self.num_workers, collate_fn=self.collate_fn,
#                                            shuffle=self.shuffle, pin_memory=self.pin_memory)

#     def val_dataloader(self, num_workers: int = 2) -> torch.utils.data.DataLoader:
#         return torch.utils.data.DataLoader(self.val_dataset, self.batch_size, num_workers=num_workers,
#                                            collate_fn=self.collate_fn, shuffle=False)

#     #########################################################################################################
#     # Processing ############################################################################################
#     #########################################################################################################

#     @abc.abstractmethod
#     def _add_edge_attributes(self, data: Data) -> Data:
#         max_value = self.hparams.get("preprocessing", {}).get("r", None)
#         edge_attr = Cartesian(norm=True, cat=False, max_value=max_value)
#         return edge_attr(data)

#     #########################################################################################################
#     # Data Loading ##########################################################################################
#     #########################################################################################################
#     def _load_dataset(self, mode: str):
#         processed_files = self.processed_files(mode)
#         logging.debug(f"Loaded dataset with {len(processed_files)} processed files")
#         return EventDataset(processed_files, load_func=self.load_processed_file)

#     def load_processed_file(self, f_path: str) -> Data:
#         data = self._load_processed_file(f_path)

#         # Post-Processing on loaded data before giving to data loader. Crop and index the bounding boxes
#         # and apply the transform if it is defined.
#         if hasattr(data, 'bbox'):
#             data.bbox = data.bbox.view(-1, 5)
#             data.bbox = crop_to_frame(data.bbox, image_shape=self.dims)
#         if self.transform is not None:
#             data = self.transform(data)

#         # Add a default edge attribute, if the data does not have them already.
#         if not hasattr(data, 'edge_attr') or data.edge_attr is None:
#             data = self._add_edge_attributes(data)

#         # Checking the loaded data for the sake of assuring shape consistency.
#         assert data.pos.shape[0] == data.x.shape[0], "x and pos not matching in length"
#         assert data.pos.shape[-1] >= 2
#         assert data.x.shape[-1] == 1
#         assert data.edge_attr.shape[0] == data.edge_index.shape[1], "edges index and attribute not matching"
#         assert data.edge_attr.shape[-1] >= 2, "wrong edge attribute dimension"
#         if hasattr(data, 'bbox'):
#             assert len(data.bbox.shape) == 2 and data.bbox.shape[1] == 5
#             assert len(data.y) == data.bbox.shape[0], "annotations not matching"

#         return data

#     @staticmethod
#     def collate_fn(data_list: List[Data]) -> torch_geometric.data.Batch:
#         batch = torch_geometric.data.Batch.from_data_list(data_list)
#         if hasattr(data_list[0], 'bbox'):
#             batch_bbox = sum([[i] * len(data.y) for i, data in enumerate(data_list)], [])
#             batch.batch_bbox = torch.tensor(batch_bbox, dtype=torch.long)
#         return batch


#     #########################################################################################################
#     # Dataset Properties ####################################################################################
#     #########################################################################################################
#     @classmethod
#     def add_argparse_args(cls, parent_parser: argparse.ArgumentParser, **kwargs) -> argparse.ArgumentParser:
#         parent_parser.add_argument("--dataset", action="store", type=str, required=True)

#         group = parent_parser.add_argument_group("Data")
#         group.add_argument("--batch-size", action="store", default=8, type=int)
#         group.add_argument("--num-workers", action="store", default=8, type=int)
#         group.add_argument("--pin-memory", action="store_true")
#         return parent_parser

#     @property
#     def root(self) -> str:
#         return os.path.join(os.environ["AEGNN_DATA_DIR"], self.__class__.__name__.lower())

#     @property
#     def name(self) -> str:
#         return self.__class__.__name__.lower()

#     @property
#     def classes(self) -> List[str]:
#         raise NotImplementedError

#     @property
#     def num_classes(self) -> int:
#         return len(self.classes)

#     def __repr__(self):
#         train_desc = self.train_dataset.__repr__()
#         val_desc = self.val_dataset.__repr__()
#         return f"{self.__class__.__name__}[Train: {train_desc}\nValidation: {val_desc}]"

# class EventDataset(Dataset):

#     def __init__(self, files: List[str], load_func: Callable[[str], Data]):
#         self.files = files
#         self.load_func = load_func

#     def __getitem__(self, index: int) -> T_co:
#         data_file = self.files[index]
#         return self.load_func(data_file)

#     def __len__(self) -> int:
#         return len(self.files)
