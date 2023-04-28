import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import elu, log_softmax
from torch_geometric.nn.conv import SplineConv, TransformerConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian

# from aegnn.models.layer import MaxPooling, MaxPoolingX
from layers import MaxPooling, MaxPoolingX
from torch_geometric.nn import SGConv
# from aegnn.asyncronous import make_model_asynchronous


class GraphRes(torch.nn.Module):

    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False):
        super(GraphRes, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])

        # Set dataset specific hyper-parameters.
        if dataset == "ncars":
            kernel_size = 2
            n = [1, 8, 16, 16, 16, 32, 32, 32, 32]
            pooling_outputs = 32
        elif dataset == "ncaltech101" or dataset == "gen1":
            kernel_size = 8
            n = [1, 16, 32, 32, 32, 128, 128, 128]
            pooling_outputs = 128
        else:
            raise NotImplementedError(f"No model parameters for dataset {dataset}")

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

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        # print(data)
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
        # print(data)
        data.x = elu(self.conv5(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm5(data.x)
        # print("TEST", data)
        data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        x_sc = data.x.clone()
        data.x = elu(self.conv6(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm6(data.x)
        data.x = elu(self.conv7(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm7(data.x)
        data.x = data.x + x_sc

        x = self.pool7(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features)
        return self.fc(x)


class SimpleNet(torch.nn.Module):
    def __init__(self, input_shape: torch.Tensor, num_outputs: int):
        super().__init__()
        self.conv1 = SGConv(1, 1, k=4,
                            cached=True)
        self.norm1 = BatchNorm(in_channels=1)
        self.pool1 = MaxPoolingX(input_shape[:2], size=256)
        self.linear = Linear(256, out_features=num_outputs, bias=True)

        # # V3
        # self.conv1 = SGConv(1, 20, k=4,
        #                     cached=True)
        # self.conv2 = SGConv(1, 40, k=4,
        #                     cached=True)
        # self.conv1 = SGConv(1, 100, k=4,
        #                     cached=True)
        # self.norm1 = BatchNorm(in_channels=1)
        # self.pool1 = MaxPoolingX(input_shape[:2], size=256)
        # self.linear = Linear(256, out_features=num_outputs, bias=True)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        # simple dimple
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # working version 1
        x = self.conv1(x, edge_index)
        x = torch.tensor_split(x, data.y.shape[0])
        x = [torch.mean(subx, dim=0, keepdim=False) for subx in x]
        x = torch.stack(x, dim=0)

        # # V2
        # data.x = self.conv1(data.x, edge_index, edge_weight=edge_attr[:,0] + edge_attr[:,1])
        # data.x = self.norm1(data.x)
        # # print("x before:", x, x.shape)
        # x = self.pool1(data.x, pos=data.pos[:, :2], batch=data.batch)
        # # print("x after:", x, x.shape)
        # # print("FEATURES:", x, x.shape)
        # x = torch.tensor_split(torch.flatten(x), 16)
        # x = [self.linear(subx) for subx in x]
        # x = torch.stack(x, dim=0)
        # # x = x.view(-1, self.linear.in_features)
        # return x

        # # V3
        # data.x = self.conv1(data.x, edge_index, edge_weight=edge_attr[:,0] + edge_attr[:,1])
        # data.x = elu(data.x)
        # data.x = self.norm1(data.x)
        # # print("x before:", x, x.shape)
        # x = self.pool1(data.x, pos=data.pos[:, :2], batch=data.batch)
        # x = self.conv1(x, edge_index, edge_weight=edge_attr[:,0] + edge_attr[:,1])
        # # print("x after:", x, x.shape)
        # # print("FEATURES:", x, x.shape)
        # x = torch.tensor_split(torch.flatten(x), 16)
        # x = [self.linear(subx) for subx in x]
        # x = torch.stack(x, dim=0)
        # # x = x.view(-1, self.linear.in_features)
        return x


        return x
    


class GraphResModified(torch.nn.Module):

    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False):
        super(GraphResModified, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])

        # Set dataset specific hyper-parameters.
        if dataset == "ncars":
            kernel_size = 2
            n = [1, 8, 16, 16, 16, 32, 32, 32, 32]
            pooling_outputs = 32
        elif dataset == "ncaltech101" or dataset == "gen1":
            kernel_size = 8
            n = [1, 16, 32, 32, 32, 128, 128, 128]
            pooling_outputs = 128
        else:
            raise NotImplementedError(f"No model parameters for dataset {dataset}")

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
        # self.conv7 = SplineConv(n[6], n[7], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.conv7 = TransformerConv(n[6], n[7], edge_dim=3, heads=6)
        self.conv8 = TransformerConv(n[7] * 6, n[7] // 2, edge_dim=3, heads=2)
        self.norm7 = BatchNorm(in_channels=n[7])

        self.pool7 = MaxPoolingX(input_shape[:2] // 4, size=16)
        self.fc = Linear(pooling_outputs * 16, out_features=num_outputs, bias=bias)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        # print(data)
        data.x = self.conv1(data.x, data.edge_index, data.edge_attr)
        data.x = elu(data.x)
        data.x = self.norm1(data.x)
        data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2(data.x)

        x_sc = data.x.clone()
        data.x = elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm3(data.x)
        data.x = elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm4(data.x)
        data.x = data.x + x_sc
        # print(data)
        data.x = elu(self.conv5(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm5(data.x)
        # print("TEST", data)
        data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        x_sc = data.x.clone()
        data.x = elu(self.conv6(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm6(data.x)
        # print("BEFORE SPLINE:", data.x, data.x.shape)
        data.x = elu(self.conv7(data.x, data.edge_index, data.edge_attr))
        data.x = self.conv8(data.x, data.edge_index, data.edge_attr)
        # print("AFTER SPLINE:", data.x, data.x.shape)
        data.x = elu(data.x)
        data.x = self.norm7(data.x)
        data.x = data.x + x_sc

        x = self.pool7(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features)
        return self.fc(x)
    
class GraphTrans(torch.nn.Module):

    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False, dropout_trans=False, heads=3):
        super(GraphTrans, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])

        # Set dataset specific hyper-parameters.
        if dataset == "ncars":
            kernel_size = 2
            n = [1, 8, 16, 16, 16, 32, 32, 32, 32]
            pooling_outputs = 32
        elif dataset == "ncaltech101" or dataset == "gen1":
            kernel_size = 8
            n = [1, 16, 32, 32, 32, 128, 128, 128]
            pooling_outputs = 128
        else:
            raise NotImplementedError(f"No model parameters for dataset {dataset}")

        h = heads
        if dropout_trans:
            drpt = 0.1
        else:
            drpt = 0

        # self.conv1 = SplineConv(n[0], n[1], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.conv1 = TransformerConv(n[0], n[1], edge_dim=dim, heads=h, dropout=drpt)
        self.norm1 = BatchNorm(in_channels=n[1] * h)
        # self.conv2 = SplineConv(n[1] * 3, n[2], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.conv2 = TransformerConv(n[1] * h, n[2], edge_dim=dim, heads=1, dropout=drpt)
        self.norm2 = BatchNorm(in_channels=n[2])

        # self.conv3 = SplineConv(n[2], n[3], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.conv3 = TransformerConv(n[2], n[3], edge_dim=dim, heads=h, dropout=drpt)
        self.norm3 = BatchNorm(in_channels=n[3] * h)
        # self.conv4 = SplineConv(n[3] * h, n[4], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.conv4 = TransformerConv(n[3] * h, n[4], edge_dim=dim, heads=1, dropout=drpt)
        self.norm4 = BatchNorm(in_channels=n[4])

        # self.conv5 = SplineConv(n[4], n[5], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.conv5 = TransformerConv(n[4], n[5], edge_dim=dim, heads=1, dropout=drpt)
        self.norm5 = BatchNorm(in_channels=n[5])
        self.pool5 = MaxPooling(pooling_size, transform=Cartesian(norm=True, cat=False))

        # self.conv6 = SplineConv(n[5], n[6], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.conv6 = TransformerConv(n[5], n[6], edge_dim=dim, heads=h, dropout=drpt)
        self.norm6 = BatchNorm(in_channels=n[6] * h)
        # self.conv7 = SplineConv(n[6] * h, n[7], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.conv7 = TransformerConv(n[6] * h, n[7], edge_dim=dim, heads=1, dropout=drpt)
        self.norm7 = BatchNorm(in_channels=n[7])

        self.pool7 = MaxPoolingX(input_shape[:2] // 4, size=16)
        self.fc = Linear(pooling_outputs * 16, out_features=num_outputs, bias=bias)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        # print(data)
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
        # print(data)
        data.x = elu(self.conv5(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm5(data.x)
        # print("TEST", data)
        data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        x_sc = data.x.clone()
        data.x = elu(self.conv6(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm6(data.x)
        data.x = elu(self.conv7(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm7(data.x)
        data.x = data.x + x_sc

        x = self.pool7(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features)
        return self.fc(x)

class GraphTransFinal(torch.nn.Module):

    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False, dropout_trans=False, heads=3):
        super(GraphTransFinal, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])

        # Set dataset specific hyper-parameters.
        if dataset == "ncars":
            kernel_size = 2
            n = [1, 8, 16, 16, 16, 32, 32, 32, 32]
            pooling_outputs = 32
        elif dataset == "ncaltech101" or dataset == "gen1":
            kernel_size = 8
            n = [1, 16, 32, 32, 32, 128, 128, 128]
            pooling_outputs = 128
        else:
            raise NotImplementedError(f"No model parameters for dataset {dataset}")

        h = heads
        if dropout_trans:
            drpt = 0.1
        else:
            drpt = 0

        self.conv1 = SplineConv(n[0], n[1], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        # self.conv1 = TransformerConv(n[0], n[1], edge_dim=dim, heads=h, dropout=drpt)
        self.norm1 = BatchNorm(in_channels=n[1] * h)
        self.conv2 = SplineConv(n[1], n[2], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        # self.conv2 = TransformerConv(n[1] * h, n[2], edge_dim=dim, heads=1, dropout=drpt)
        self.norm2 = BatchNorm(in_channels=n[2])

        self.conv3 = SplineConv(n[2], n[3], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        # self.conv3 = TransformerConv(n[2], n[3], edge_dim=dim, heads=h, dropout=drpt)
        self.norm3 = BatchNorm(in_channels=n[3] * h)
        self.conv4 = SplineConv(n[3], n[4], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        # self.conv4 = TransformerConv(n[3] * h, n[4], edge_dim=dim, heads=1, dropout=drpt)
        self.norm4 = BatchNorm(in_channels=n[4])

        self.conv5 = SplineConv(n[4], n[5], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        # self.conv5 = TransformerConv(n[4], n[5], edge_dim=dim, heads=1, dropout=drpt)
        self.norm5 = BatchNorm(in_channels=n[5])
        self.pool5 = MaxPooling(pooling_size, transform=Cartesian(norm=True, cat=False))

        # self.conv6 = SplineConv(n[5], n[6], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.conv6 = TransformerConv(n[5], n[6], edge_dim=dim, heads=h, dropout=drpt)
        self.norm6 = BatchNorm(in_channels=n[6] * h)
        # self.conv7 = SplineConv(n[6] * h, n[7], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.conv7 = TransformerConv(n[6] * h, n[7], edge_dim=dim, heads=1, dropout=drpt)
        self.norm7 = BatchNorm(in_channels=n[7])

        self.pool7 = MaxPoolingX(input_shape[:2] // 4, size=16)
        self.fc = Linear(pooling_outputs * 16, out_features=num_outputs, bias=bias)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        # print(data)
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
        # print(data)
        data.x = elu(self.conv5(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm5(data.x)
        # print("TEST", data)
        data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        x_sc = data.x.clone()
        data.x = elu(self.conv6(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm6(data.x)
        data.x = elu(self.conv7(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm7(data.x)
        data.x = data.x + x_sc

        x = self.pool7(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features)
        return self.fc(x)

class GraphResSimple(torch.nn.Module):

    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False):
        super(GraphResSimple, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])

        # Set dataset specific hyper-parameters.
        kernel_size = 8
        n = [1, 16, 32, 32, 32, 128, 128, 128]
        pooling_outputs = 128

        # self.conv1 = SplineConv(n[0], n[1], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        # self.norm1 = BatchNorm(in_channels=n[1])
        # self.conv2 = SplineConv(n[1], n[2], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.conv1 = SGConv(n[0], n[2], k=4,
                            cached=True)
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
        # self.conv7 = SplineConv(n[6], n[7], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.conv7 = TransformerConv(n[6], n[7], edge_dim=3, heads=6)
        self.conv8 = TransformerConv(n[7] * 6, n[7] // 2, edge_dim=3, heads=2)
        self.norm7 = BatchNorm(in_channels=n[7])

        self.pool7 = MaxPoolingX(input_shape[:2] // 4, size=16)
        self.fc = Linear(pooling_outputs * 16, out_features=num_outputs, bias=bias)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        # print(data)
        inverse_l2len = torch.div(1, torch.norm(data.edge_attr, dim=1))

        # data.x = self.conv1(data.x, data.edge_index, data.edge_attr)
        data.x = self.conv1(data.x, data.edge_index, edge_weight=inverse_l2len)
        data.x = elu(data.x)
        data.x = self.norm2(data.x)

        x_sc = data.x.clone()
        data.x = elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm3(data.x)
        data.x = elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm4(data.x)
        data.x = data.x + x_sc
        # print(data)
        data.x = elu(self.conv5(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm5(data.x)
        # print("TEST", data)
        data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        x_sc = data.x.clone()
        data.x = elu(self.conv6(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm6(data.x)
        # print("BEFORE SPLINE:", data.x, data.x.shape)
        data.x = elu(self.conv7(data.x, data.edge_index, data.edge_attr))
        data.x = self.conv8(data.x, data.edge_index, data.edge_attr)
        # print("AFTER SPLINE:", data.x, data.x.shape)
        data.x = elu(data.x)
        data.x = self.norm7(data.x)
        data.x = data.x + x_sc

        x = self.pool7(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features)
        return self.fc(x)
    
class GraphResSimple2(torch.nn.Module): # this time three more splines were replaced

    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False):
        super(GraphResSimple, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])

        # Set dataset specific hyper-parameters.
        kernel_size = 8
        n = [1, 16, 32, 32, 32, 128, 128, 128]
        pooling_outputs = 128

        # self.conv1 = SplineConv(n[0], n[1], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        # self.norm1 = BatchNorm(in_channels=n[1])
        # self.conv2 = SplineConv(n[1], n[2], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.conv1 = SGConv(n[0], n[4], k=4,
                            cached=True)
        # self.norm2 = BatchNorm(in_channels=n[2])

        # self.conv3 = SplineConv(n[2], n[3], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        # self.norm3 = BatchNorm(in_channels=n[3])
        # self.conv4 = SplineConv(n[3], n[4], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm4 = BatchNorm(in_channels=n[4])

        self.conv5 = SplineConv(n[4], n[5], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm5 = BatchNorm(in_channels=n[5])
        self.pool5 = MaxPooling(pooling_size, transform=Cartesian(norm=True, cat=False))

        self.conv6 = SplineConv(n[5], n[6], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm6 = BatchNorm(in_channels=n[6])
        # self.conv7 = SplineConv(n[6], n[7], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.conv7 = TransformerConv(n[6], n[7], edge_dim=3, heads=6)
        self.conv8 = TransformerConv(n[7] * 6, n[7] // 2, edge_dim=3, heads=2)
        self.norm7 = BatchNorm(in_channels=n[7])

        self.pool7 = MaxPoolingX(input_shape[:2] // 4, size=16)
        self.fc = Linear(pooling_outputs * 16, out_features=num_outputs, bias=bias)

    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:
        # print(data)
        inverse_l2len = torch.div(1, torch.norm(data.edge_attr, dim=1))

        # data.x = self.conv1(data.x, data.edge_index, data.edge_attr)
        data.x = self.conv1(data.x, data.edge_index, edge_weight=inverse_l2len)
        data.x = elu(data.x)
        # data.x = self.norm2(data.x)

        # x_sc = data.x.clone()
        # data.x = elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        # data.x = self.norm3(data.x)
        # data.x = elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm4(data.x)
        # data.x = data.x + x_sc
        data.x = data.x
        # print(data)
        data.x = elu(self.conv5(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm5(data.x)
        # print("TEST", data)
        data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        x_sc = data.x.clone()
        data.x = elu(self.conv6(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm6(data.x)
        # print("BEFORE SPLINE:", data.x, data.x.shape)
        data.x = elu(self.conv7(data.x, data.edge_index, data.edge_attr))
        data.x = self.conv8(data.x, data.edge_index, data.edge_attr)
        # print("AFTER SPLINE:", data.x, data.x.shape)
        data.x = elu(data.x)
        data.x = self.norm7(data.x)
        data.x = data.x + x_sc

        x = self.pool7(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features)
        return self.fc(x)