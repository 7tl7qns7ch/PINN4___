import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import get_xytgrid, get_3dboundary, get_3dboundary_points
from train_utils.utils import vor2vel, torch2dgrid
import scipy.io
import h5py


class NSdata(object):
    def __init__(self, datapath1,
                 nx, nt,
                 offset=0, num=1,
                 datapath2=None,
                 sub=1, sub_t=1,
                 vel=False, t_interval=1.0):
        '''
        Load data from npy and reshape to (N, X, Y, T)
        Args:
            datapath1: path to data
            nx: number of points in each spatial domain
            nt: number of points in temporal domain
            offset: index of the instance
            num: number of instances
            datapath2: path to second part of data, default None
            sub: downsample interval of spatial domain
            sub_t: downsample interval of temporal domain
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

        # transpose data into (N, S, S, T)
        part1 = data1.permute(0, 2, 3, 1)
        if datapath2 is not None:
            part2 = data2.permute(0, 2, 3, 1)
            self.data = torch.cat((part1, part2), dim=0)
        else:
            self.data = part1
        self.vor = self.data[offset: offset + num, :, :, :].cpu()
        if vel:
            self.vel_u, self.vel_v = vor2vel(self.vor)  # Compute velocity from vorticity

    def get_init_cond(self):
        values = np.stack([self.vel_u[0, :, :, 0],
                           self.vel_v[0, :, :, 0],
                           self.vor[0, :, :, 0]], axis=2)
        return values

    def get_boundary_value(self, component=0):
        '''
        Get the boundary value for component-th output
        Args:
            component: int, 0: velocity_u; 1: velocity_v; 2: vorticity;
        Returns:
            value: N by 1 array, boundary value of the component
        '''
        if component == 0:
            value = self.vel_u
        elif component == 1:
            value = self.vel_v
        elif component == 2:
            value = self.vor
        else:
            raise ValueError(f'No component {component} ')

        boundary = get_3dboundary(value)
        return boundary

    def get_boundary_points(self, num_x, num_y, num_t):
        '''
        Args:
            num_x:
            num_y:

        Returns:
            points: N by 3 array
        '''
        points = get_3dboundary_points(num_x, num_y, num_t,
                                       bot=(0, 0, 0),
                                       top=(2 * np.pi, 2 * np.pi, self.time_scale))
        return points

    def get_init_cond(self):
        values = np.stack([self.vel_u[0, :, :, 0],
                           self.vel_v[0, :, :, 0],
                           self.vor[0, :, :, 0]], axis=2)
        return values

    def get_test_xyt(self):
        '''

        Returns:
            points: (x, y, t) array with shape (S * S * T, 3)
            values: (u, v, w) array with shape (S * S * T, 3)

        '''
        points = get_xytgrid(S=self.S, T=self.T,
                             bot=[0, 0, 0],
                             top=[2 * np.pi, 2 * np.pi, self.time_scale])
        u_val = np.ravel(self.vel_u)
        v_val = np.ravel(self.vel_v)
        w_val = np.ravel(self.vor)
        values = np.stack([u_val, v_val, w_val], axis=0).T
        return points, values

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
    

class NSLong(object):
    def __init__(self,
                 datapath,
                 nx, nt,
                 time_scale,
                 offset=0,
                 num=1, vel=False):
        '''
        Load data from mat
        Args:
            datapath: path to data file
            nx: number of points in each spatial domain
            nt: number of points in temporal domain
            offset: index of the instance
            num: number of instances
            vel: compute velocity from vorticity if True
        '''

        self.time_scale = time_scale
        self.S = nx
        self.T = nt

        with h5py.File(datapath, mode='r') as file:
            raw = file['u']
            data = np.array(raw)
        vor = torch.tensor(data, dtype=torch.float).permute(3, 1, 2, 0)
        self.vor = vor[offset: offset + num, :, :, :]     # num x 64 x 64 x 50
        if vel:
            self.vel_u, self.vel_v = vor2vel(self.vor, L=1.0)

    def get_boundary_value(self, component=0):
        '''
            Get the boundary value for component-th output
            Args:
                component: int, 0: velocity_u; 1: velocity_v; 2: vorticity;
            Returns:
                value: N by 1 array, boundary value of the component
        '''
        if component == 0:
            value = self.vel_u
        elif component == 1:
            value = self.vel_v
        elif component == 2:
            value = self.vor
        else:
            raise ValueError(f'No component {component} ')

        boundary = get_3dboundary(value)
        return boundary

    def get_boundary_points(self, num_x, num_y, num_t):
        points = get_3dboundary_points(num_x, num_y, num_t,
                                       bot=(0,0,0),
                                       top=(1, 1, self.time_scale))
        return points

    def get_test_xyt(self):
        '''

        Returns:
            points: (x, y, t) array with shape (S * S * T, 3)
            values: (u, v, w) array with shape (S * S * T, 3)

        '''
        points = get_xytgrid(S=self.S, T=self.T,
                             bot=[0, 0, 0],
                             top=[1, 1, self.time_scale])
        u_val = np.ravel(self.vel_u)
        v_val = np.ravel(self.vel_v)
        w_val = np.ravel(self.vor)
        values = np.stack([u_val, v_val, w_val], axis=0).T
        return points, values
