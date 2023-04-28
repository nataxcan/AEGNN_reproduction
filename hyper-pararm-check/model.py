import torch
import torch_geometric
import pytorch_lightning as pl


from torch.nn.functional import softmax
from typing import Tuple

import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import elu
from torch_geometric.nn.conv import SplineConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian

from typing import Callable, Dict, List, Optional, Union

# from aegnn.models.layer import MaxPooling, MaxPoolingX
from layers import MaxPooling, MaxPoolingX
# from aegnn.asyncronous import make_model_asynchronous


class GraphRes(torch.nn.Module):

    def __init__(self, hiddenLayers, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False):
        super(GraphRes, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])
        self.hid = hiddenLayers
        # Set dataset specific hyper-parameters.
        if dataset == "ncars":
            kernel_size = 2
            n = [1, 8, 16, 16, 16, 32, 32, 32, 32]
            pooling_outputs = 32
        elif dataset == "ncaltech101" or dataset == "gen1":
            kernels = [
                 [1, 32, 128],
                 [1, 16, 32, 128, 128],
                 [1, 16, 32, 32, 32, 128, 128, 128],
                 [1, 16, 16, 32, 32, 32, 128, 128, 128, 128, 128]
                ]
            n = kernels[hiddenLayers]
            kernel_size = len(n)
            
            pooling_outputs = 128
        else:
            raise NotImplementedError(f"No model parameters for dataset {dataset}")
        
        if kernel_size == 3:

            self.conv1 = SplineConv(n[0], n[1], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm1 = BatchNorm(in_channels=n[1])
            self.conv2 = SplineConv(n[1], n[2], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm2 = BatchNorm(in_channels=n[2])
            self.pool2 = MaxPoolingX(input_shape[:2] // 4, size=16)
            self.fc = Linear(pooling_outputs * 16, out_features=num_outputs, bias=bias)

        if kernel_size == 5:

            self.conv1 = SplineConv(n[0], n[1], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm1 = BatchNorm(in_channels=n[1])
            self.conv2 = SplineConv(n[1], n[2], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm2 = BatchNorm(in_channels=n[2])
            self.conv3 = SplineConv(n[2], n[3], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm3 = BatchNorm(in_channels=n[3])

            self.pool3 = MaxPooling(pooling_size, transform=Cartesian(norm=True, cat=False))

            self.conv4 = SplineConv(n[3], n[4], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm4 = BatchNorm(in_channels=n[4])

            self.pool4 = MaxPoolingX(input_shape[:2] // 4, size=16)
            self.fc = Linear(pooling_outputs * 16, out_features=num_outputs, bias=bias)

        if kernel_size == 8:

            self.conv1 = SplineConv(n[0], n[1], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm1 = BatchNorm(in_channels=n[1])
            self.conv2 = SplineConv(n[1], n[2], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm2 = BatchNorm(in_channels=n[2])

            self.conv3 = SplineConv(n[2], n[3], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm3 = BatchNorm(in_channels=n[3])
            self.conv4 = SplineConv(n[3], n[4], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm4 = BatchNorm(in_channels=n[4])

            self.conv5 = SplineConv(n[4], n[5], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm5 = BatchNorm(in_channels=n[5])
            self.pool5 = MaxPooling(pooling_size, transform=Cartesian(norm=True, cat=False))

            self.conv6 = SplineConv(n[5], n[6], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm6 = BatchNorm(in_channels=n[6])
            self.conv7 = SplineConv(n[6], n[7], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm7 = BatchNorm(in_channels=n[7])

            self.pool7 = MaxPoolingX(input_shape[:2] // 4, size=16)
            self.fc = Linear(pooling_outputs * 16, out_features=num_outputs, bias=bias)

        if kernel_size == 11:

            self.conv1 = SplineConv(n[0], n[1], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm1 = BatchNorm(in_channels=n[1])
            self.conv2 = SplineConv(n[1], n[2], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm2 = BatchNorm(in_channels=n[2])

            self.conv3 = SplineConv(n[2], n[3], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm3 = BatchNorm(in_channels=n[3])

            self.conv4 = SplineConv(n[3], n[4], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm4 = BatchNorm(in_channels=n[4])
            self.conv5 = SplineConv(n[4], n[5], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm5 = BatchNorm(in_channels=n[5])

            self.conv6 = SplineConv(n[5], n[6], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm6 = BatchNorm(in_channels=n[6])


            self.conv7 = SplineConv(n[6], n[7], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm7 = BatchNorm(in_channels=n[7])
            self.pool7 = MaxPooling(pooling_size, transform=Cartesian(norm=True, cat=False))
            self.conv8 = SplineConv(n[7], n[8], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm8 = BatchNorm(in_channels=n[8])
            self.conv9 = SplineConv(n[8], n[9], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm9 = BatchNorm(in_channels=n[9])
            self.conv10 = SplineConv(n[9], n[10], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
            self.norm10 = BatchNorm(in_channels=n[10])

            self.pool10 = MaxPoolingX(input_shape[:2] // 4, size=16)
            self.fc = Linear(pooling_outputs * 16, out_features=num_outputs, bias=bias)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        if self.hid == 0:
            data.x = elu(self.conv1(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm1(data.x)
            data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm2(data.x)
            x = self.pool2(data.x, pos=data.pos[:, :2], batch=data.batch)
            x = x.view(-1, self.fc.in_features)

        if self.hid == 1:
            data.x = elu(self.conv1(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm1(data.x)
            data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm2(data.x)
            x_sc = data.x.clone()
            data.x = elu(self.conv3(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm3(data.x)
            data = self.pool3(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)
            data.x = elu(self.conv4(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm4(data.x)
            x = self.pool4(data.x, pos=data.pos[:, :2], batch=data.batch)
            x = x.view(-1, self.fc.in_features)

        if self.hid == 2:
            data.x = elu(self.conv1(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm1(data.x)
            data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm2(data.x)

            x_sc = data.x.clone()
            data.x = elu(self.conv3(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm3(data.x)
            data.x = elu(self.conv4(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm4(data.x)
            data.x = data.x + x_sc
            data.x = elu(self.conv5(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm5(data.x)
            data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

            x_sc = data.x.clone()
            data.x = elu(self.conv6(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm6(data.x)
            data.x = elu(self.conv7(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm7(data.x)
            data.x = data.x + x_sc

            x = self.pool7(data.x, pos=data.pos[:, :2], batch=data.batch)
            x = x.view(-1, self.fc.in_features)

        if self.hid == 3:
            data.x = elu(self.conv1(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm1(data.x)
            data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm2(data.x)

            x_sc = data.x.clone()

            data.x = elu(self.conv3(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm3(data.x)
            data.x = elu(self.conv4(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm4(data.x)

            

            data.x = elu(self.conv5(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm5(data.x)
            # data.x = data.x + x_sc
            data.x = elu(self.conv6(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm6(data.x)

        
            x_sc = data.x.clone()
            data = self.pool7(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

            data.x = elu(self.conv7(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm7(data.x)
            data.x = elu(self.conv8(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm8(data.x)

            # data.x = data.x + x_sc

            data.x = elu(self.conv9(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm9(data.x)
            data.x = elu(self.conv10(data.x, data.edge_index, data.edge_attr))
            data.x = self.norm10(data.x)

            x = self.pool10(data.x, pos=data.pos[:, :2], batch=data.batch)
            x = x.view(-1, self.fc.in_features)

        return self.fc(x)
