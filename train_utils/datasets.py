import scipy.io
import numpy as np
from argparse import ArgumentParser
import yaml
from torch.utils.data import DataLoader

try:
    from pyDOE import lhs
    # Only needed for PINN's dataset
except ImportError:
    lhs = None

import torch
from torch.utils.data import Dataset
from .utils import get_grid3d #, convert_ic, torch2dgrid


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class NSLoader(object):
    def __init__(self, datapath1,
                 nx, nt,
                 datapath2=None, sub=1, sub_t=1,
                 N=100, t_interval=1.0):
        '''
        Load data from npy and reshape to (N, X, Y, T)
        Args:
            datapath1: path to data
            nx:
            nt:
            datapath2: path to second part of data, default None
            sub:
            sub_t:
            N:
            t_interval:
        '''
        self.S = nx // sub
        self.T = int(nt * t_interval) // sub_t + 1
        self.time_scale = t_interval
        data1 = np.load(datapath1)
        data1 = torch.tensor(data1, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]

        if datapath2 is not None:
            data2 = np.load(datapath2)
            data2 = torch.tensor(data2, dtype=torch.float)[..., ::sub_t, ::sub, ::sub]
        if t_interval == 0.5:
            data1 = self.extract(data1)
            if datapath2 is not None:
                data2 = self.extract(data2)
        part1 = data1.permute(0, 2, 3, 1)
        if datapath2 is not None:
            part2 = data2.permute(0, 2, 3, 1)
            self.data = torch.cat((part1, part2), dim=0)
        else:
            self.data = part1

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        if train:
            a_data = self.data[start:start + n_sample, :, :, 0].reshape(n_sample, self.S, self.S)
            u_data = self.data[start:start + n_sample].reshape(n_sample, self.S, self.S, self.T)
        else:
            a_data = self.data[-n_sample:, :, :, 0].reshape(n_sample, self.S, self.S)
            u_data = self.data[-n_sample:].reshape(n_sample, self.S, self.S, self.T)
        a_data = a_data.reshape(n_sample, self.S, self.S, 1, 1).repeat([1, 1, 1, self.T, 1])
        gridx, gridy, gridt = get_grid3d(self.S, self.T, time_scale=self.time_scale)
        a_data = torch.cat((
            gridx.repeat([n_sample, 1, 1, 1, 1]), 
            gridy.repeat([n_sample, 1, 1, 1, 1]),
            gridt.repeat([n_sample, 1, 1, 1, 1]), 
            a_data), dim=-1)
        dataset = torch.utils.data.TensorDataset(a_data, u_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
        return loader

    def make_dataset(self, n_sample, start=0, train=True):
        if train:
            a_data = self.data[start:start + n_sample, :, :, 0].reshape(n_sample, self.S, self.S)
            u_data = self.data[start:start + n_sample].reshape(n_sample, self.S, self.S, self.T)
        else:
            a_data = self.data[-n_sample:, :, :, 0].reshape(n_sample, self.S, self.S)
            u_data = self.data[-n_sample:].reshape(n_sample, self.S, self.S, self.T)
        a_data = a_data.reshape(n_sample, self.S, self.S, 1, 1).repeat([1, 1, 1, self.T, 1])
        gridx, gridy, gridt = get_grid3d(self.S, self.T)
        a_data = torch.cat((
            gridx.repeat([n_sample, 1, 1, 1, 1]),
            gridy.repeat([n_sample, 1, 1, 1, 1]),
            gridt.repeat([n_sample, 1, 1, 1, 1]),
            a_data), dim=-1)
        dataset = torch.utils.data.TensorDataset(a_data, u_data)
        return dataset

    @staticmethod
    def extract(data):
        '''
        Extract data with time range 0-0.5, 0.25-0.75, 0.5-1.0, 0.75-1.25,...
        Args:
            data: tensor with size N x 129 x 128 x 128

        Returns:
            output: (4*N-1) x 65 x 128 x 128
        '''
        T = data.shape[1] // 2
        interval = data.shape[1] // 4
        N = data.shape[0]
        new_data = torch.zeros(4 * N - 1, T + 1, data.shape[2], data.shape[3])
        for i in range(N):
            for j in range(4):
                if i == N - 1 and j == 3:
                    # reach boundary
                    break
                if j != 3:
                    new_data[i * 4 + j] = data[i, interval * j:interval * j + T + 1]
                else:
                    new_data[i * 4 + j, 0: interval] = data[i, interval * j:interval * j + interval]
                    new_data[i * 4 + j, interval: T + 1] = data[i + 1, 0:interval + 1]
        return new_data


class KFDataset(Dataset):
    def __init__(self, paths, 
                 data_res, 
                 pde_res, 
                 raw_res, 
                 n_samples=None, 
                 total_samples=None,
                 idx=0,
                 offset=0,
                 t_duration=1.0):
        super().__init__()
        self.data_res = data_res    # data resolution
        self.pde_res = pde_res      # pde loss resolution
        self.raw_res = raw_res      # raw data resolution
        self.t_duration = t_duration
        self.paths = paths
        self.offset = offset
        self.n_samples = n_samples
        if t_duration == 1.0:
            self.T = self.pde_res[2]
        else:
            self.T = int(self.pde_res[2] * t_duration) + 1    # number of points in time dimension

        self.load()
        if total_samples is not None:
            print(f'Load {total_samples} samples starting from {idx}th sample')
            self.data = self.data[idx:idx + total_samples]
            self.a_data = self.a_data[idx:idx + total_samples]
            
        self.data_s_step = pde_res[0] // data_res[0]
        self.data_t_step = (pde_res[2] - 1) // (data_res[2] - 1)

        print(self.data.shape)               # N x S x S x T
        print(self.a_data.shape)             # N x S x S x 1 x 1

    def load(self):
        datapath = self.paths[0]
        raw_data = np.load(datapath, mmap_mode='r')
        # print(raw_data.shape)

        # subsample ratio
        sub_x = self.raw_res[0] // self.data_res[0]
        sub_t = (self.raw_res[2] - 1) // (self.data_res[2] - 1)
        a_sub_x = self.raw_res[0] // self.pde_res[0]
        
        # load data
        data = raw_data[self.offset: self.offset + self.n_samples, ::sub_t, ::sub_x, ::sub_x]
        # print(data.shape)
        
        # divide data
        if self.t_duration != 0.:
            end_t = self.raw_res[2] - 1
            K = int(1 / self.t_duration)
            step = end_t // K
            data = self.partition(data)
            a_data = raw_data[self.offset: self.offset + self.n_samples, 0:end_t:step, ::a_sub_x, ::a_sub_x]
            a_data = a_data.reshape(self.n_samples * K, 1, self.pde_res[0], self.pde_res[1])    # 2N x 1 x S x S

        else:
            a_data = raw_data[self.offset: self.offset + self.n_samples, 0:1, ::a_sub_x, ::a_sub_x]

        # convert into torch tensor
        data = torch.from_numpy(data).to(torch.float32)
        a_data = torch.from_numpy(a_data).to(torch.float32).permute(0, 2, 3, 1)
        self.data = data.permute(0, 2, 3, 1)

        S = self.pde_res[1]
        
        a_data = a_data[:, :, :, :, None]   # N x S x S x 1 x 1
        gridx, gridy, gridt = get_grid3d(S, self.T)
        self.grid = torch.cat((gridx[0], gridy[0], gridt[0]), dim=-1)   # S x S x T x 3
        self.a_data = a_data
        # print(self.data.shape)               # N x S x S x T
        # print(self.a_data.shape)             # N x S x S x 1 x 1

    def partition(self, data):
        '''
        Args:
            data: tensor with size N x T x S x S

        Returns:
            output: int(1/t_duration) *N x (T//2 + 1) x 128 x 128
        '''
        N, T, S = data.shape[:3]
        K = int(1 / self.t_duration)
        new_data = np.zeros((K * N, T // K + 1, S, S))
        step = T // K
        for i in range(N):
            for j in range(K):
                new_data[i * K + j] = data[i, j * step: (j + 1) * step + 1]
        return new_data
    
    def __getitem__(self, idx):
        a_data = torch.cat((self.grid, self.a_data[idx].repeat(1, 1, self.T, 1)), dim=-1)    # repeat T
        return self.data[idx], a_data               # N x S x S x T, N x S x S x T x 4

    def __len__(self, ):
        return self.data.shape[0]
    

''' dataset class for loading initial conditions for Komogrov flow '''
class KFaDataset(Dataset):
    def __init__(self, paths, 
                 pde_res, 
                 raw_res, 
                 n_samples=None, 
                 offset=0,
                 t_duration=1.0):
        super().__init__()
        self.pde_res = pde_res      # pde loss resolution
        self.raw_res = raw_res      # raw data resolution
        self.t_duration = t_duration
        self.paths = paths
        self.offset = offset
        self.n_samples = n_samples
        if t_duration == 1.0:
            self.T = self.pde_res[2]
        else:
            self.T = int(self.pde_res[2] * t_duration) + 1    # number of points in time dimension

        self.load()

    def load(self):
        datapath = self.paths[0]
        raw_data = np.load(datapath, mmap_mode='r')
        # subsample ratio
        a_sub_x = self.raw_res[0] // self.pde_res[0]
        # load data
        if self.t_duration != 0.:
            end_t = self.raw_res[2] - 1
            K = int(1/self.t_duration)
            step = end_t // K
            a_data = raw_data[self.offset: self.offset + self.n_samples, 0:end_t:step, ::a_sub_x, ::a_sub_x]
            a_data = a_data.reshape(self.n_samples * K, 1, self.pde_res[0], self.pde_res[1])    # 2N x 1 x S x S
        else:
            a_data = raw_data[self.offset: self.offset + self.n_samples, 0:1, ::a_sub_x, ::a_sub_x]

        # convert into torch tensor
        a_data = torch.from_numpy(a_data).to(torch.float32).permute(0, 2, 3, 1)
        S = self.pde_res[1]
        a_data = a_data[:, :, :, :, None]   # N x S x S x 1 x 1
        gridx, gridy, gridt = get_grid3d(S, self.T)
        self.grid = torch.cat((gridx[0], gridy[0], gridt[0]), dim=-1)   # S x S x T x 3
        self.a_data = a_data

    def __getitem__(self, idx):
        a_data = torch.cat((self.grid, self.a_data[idx].repeat(1, 1, self.T, 1)), dim=-1)
        return a_data

    def __len__(self, ):
        return self.a_data.shape[0]


if __name__ == "__main__":

    from utils import get_grid3d #, convert_ic, torch2dgrid

    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config', type=str, default="pino/configs/Re500-05s-test.yaml", help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)

    batchsize = config['train']['batchsize']
    dataset = KFDataset(paths=config['data']['paths'], 
                        raw_res=config['data']['raw_res'],
                        data_res=config['data']['data_res'], 
                        pde_res=config['data']['data_res'], 
                        n_samples=config['data']['n_sample'], 
                        total_samples=1,
                        idx=0,
                        offset=config['data']['testoffset'], 
                        t_duration=1)
    
    # print(dataset[0][0].shape, dataset[0][1].shape)  # torch.Size([256, 256, 129]) torch.Size([256, 256, 129, 4])
    