import numpy as np

import torch
import torch.autograd as autograd


def weighted_mse(pred, target, weight=None):
    if weight is None:
        return torch.mean((pred - target) ** 2)
    else:
        return torch.mean(weight * (pred - target) ** 2)
    

def get_3dboundary_points(num_x,                # number of points on x axis
                          num_y,                # number of points on y axis
                          num_t,                # number of points on t axis
                          bot=(0, 0, 0),        # lower bound
                          top=(1.0, 1.0, 1.0)   # upper bound
                          ):
    x_top, y_top, t_top = top
    x_bot, y_bot, t_bot = bot

    x_arr = np.linspace(x_bot, x_top, num=num_x, endpoint=False)
    y_arr = np.linspace(y_bot, y_top, num=num_y, endpoint=False)
    xx, yy = np.meshgrid(x_arr, y_arr, indexing='ij')
    xarr = np.ravel(xx)
    yarr = np.ravel(yy)
    tarr = np.ones_like(xarr) * t_bot
    point0 = np.stack([xarr, yarr, tarr], axis=0).T  # (SxSx1, 3), boundary on t=0

    t_arr = np.linspace(t_bot, t_top, num=num_t)
    yy, tt = np.meshgrid(y_arr, t_arr, indexing='ij')
    yarr = np.ravel(yy)
    tarr = np.ravel(tt)
    xarr = np.ones_like(yarr) * x_bot
    point2 = np.stack([xarr, yarr, tarr], axis=0).T  # (1xSxT, 3), boundary on x=0

    xarr = np.ones_like(yarr) * x_top
    point3 = np.stack([xarr, yarr, tarr], axis=0).T  # (1xSxT, 3), boundary on x=2pi

    xx, tt = np.meshgrid(x_arr, t_arr, indexing='ij')
    xarr = np.ravel(xx)
    tarr = np.ravel(tt)
    yarr = np.ones_like(xarr) * y_bot
    point4 = np.stack([xarr, yarr, tarr], axis=0).T  # (128x1x65, 3), boundary on y=0

    yarr = np.ones_like(xarr) * y_top
    point5 = np.stack([xarr, yarr, tarr], axis=0).T  # (128x1x65, 3), boundary on y=2pi

    points = np.concatenate([point0,
                             point2, point3,
                             point4, point5],
                            axis=0)
    return points


def get_3dboundary(value):
    boundary0 = value[0, :, :, 0:1]  # 128x128x1, boundary on t=0
    # boundary1 = value[0, :, :, -1:]     # 128x128x1, boundary on t=0.5
    boundary2 = value[0, 0:1, :, :]  # 1x128x65, boundary on x=0
    boundary3 = value[0, -1:, :, :]  # 1x128x65, boundary on x=1
    boundary4 = value[0, :, 0:1, :]  # 128x1x65, boundary on y=0
    boundary5 = value[0, :, -1:, :]  # 128x1x65, boundary on y=1

    part0 = np.ravel(boundary0)
    # part1 = np.ravel(boundary1)
    part2 = np.ravel(boundary2)
    part3 = np.ravel(boundary3)
    part4 = np.ravel(boundary4)
    part5 = np.ravel(boundary5)
    boundary = np.concatenate([part0,
                               part2, part3,
                               part4, part5],
                              axis=0)[:, np.newaxis]
    return boundary


def get_xytgrid(S, T, bot=[0, 0, 0], top=[1, 1, 1]):
    '''
    Args:
        S: number of points on each spatial domain
        T: number of points on temporal domain including endpoint
        bot: list or tuple, lower bound on each dimension
        top: list or tuple, upper bound on each dimension

    Returns:
        (S * S * T, 3) array
    '''
    x_arr = np.linspace(bot[0], top[0], num=S, endpoint=False)
    y_arr = np.linspace(bot[1], top[1], num=S, endpoint=False)
    t_arr = np.linspace(bot[2], top[2], num=T)
    xgrid, ygrid, tgrid = np.meshgrid(x_arr, y_arr, t_arr, indexing='ij')
    xaxis = np.ravel(xgrid)
    yaxis = np.ravel(ygrid)
    taxis = np.ravel(tgrid)
    points = np.stack([xaxis, yaxis, taxis], axis=0).T
    return points
