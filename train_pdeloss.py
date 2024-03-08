import os
import yaml
import random
from argparse import ArgumentParser
import math
from tqdm import tqdm

import torch

from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

# from models import FNO3d
from models import *

from train_utils.losses import LpLoss
from train_utils.utils import save_ckpt, count_params, dict2str


class StdScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def scale(self):
        return self.std
    

# TODO: here fill the vorticity residual
def vorticity_residual(w, re=1000.0, dt=1/32):
    # x [b t h w]
    batchsize = w.size(0)
    w = w.clone()
    w.requires_grad_(True)
    nx = w.size(2)
    ny = w.size(3)
    device = w.device

    w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])

    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1, N).reshape(1, 1, N, N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N, 1).reshape(1, 1, N, N)

    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap

    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])
    v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])
    wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])
    wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])
    wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])
    advection = u * wx + v * wy

    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)

    # establish forcing term
    x = torch.linspace(0, 2 * np.pi, nx + 1, device=device)
    x = x[0:-1]
    X, Y = torch.meshgrid(x, x)
    f = -4 * torch.cos(4 * Y)

    residual = wt + (advection - (1.0 / re) * wlap + 0.1 * w[:, 1:-1]) - f
    residual_loss = (residual ** 2).mean()
    
    return residual_loss


def load_data(data_dir):
    raw_data = np.load(data_dir).astype(np.float32)

    in_data = raw_data[-4:, ...].copy()
    out_data = raw_data[-4:, ...].copy()

    data_mean, data_scale = np.mean(out_data), np.std(out_data)

    in_data = torch.as_tensor(in_data, dtype=torch.float32)
    out_data = torch.as_tensor(out_data, dtype=torch.float32)

    flattened_in_data = []
    flattened_out_data = []

    for i in range(out_data.shape[0]):
        for j in range(out_data.shape[1] - 2):
            flattened_out_data.append(out_data[i, j:j + 3, ...])
            flattened_in_data.append(in_data[i, j:j + 3, ...])
    
    flattened_out_data = torch.stack(flattened_out_data, dim=0)
    flattened_in_data = torch.stack(flattened_in_data, dim=0)

    return flattened_out_data, flattened_in_data, data_mean.item(), data_scale.item()


def train_ns(
    model,
    test_loader,      # training data
    optimizer,
    scheduler,
    device, 
    args):

    re = args.re
    dt = args.dt
    save_step = args.save_step

    data_weight = args.data_weight
    pde_weight = args.pde_weight

    # set up directory
    base_dir = os.path.join('exp', args.logdir)
    ckpt_dir = os.path.join(base_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    # loss fn
    lploss = LpLoss(size_average=True)

    pbar = range(args.num_iter)
    if args.tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.2)

    for e in pbar:

        log_dict = {}

        optimizer.zero_grad()

        data_loss_tot = 0
        pde_loss_tot = 0
        loss_tot = 0

        for batch_index, (i_data, o_data) in enumerate(test_loader):
            # print(batch_index)

            i_data, o_data = i_data.to(device), o_data.to(device)

            # data loss
            if data_weight > 0:
                pred = model(i_data)
                data_loss = lploss(pred, o_data)
            else:
                data_loss = torch.zeros(1, device=device)

            # pde loss
            if pde_weight > 0:
                pde_loss = vorticity_residual(pred, re=re, dt=dt).detach()
            else:
                pde_loss = torch.zeros(1, device=device)

            # data loss + pde loss
            loss = data_weight * data_loss + pde_weight * pde_loss
        
            # loss backward
            loss.backward()
            optimizer.step()
            scheduler.step()

            data_loss_tot += data_loss
            pde_loss_tot += pde_loss
            loss_tot += loss

        log_dict['data'] = data_loss_tot.item() / len(test_loader)
        log_dict['pde'] = pde_loss_tot.item() / len(test_loader)
        log_dict['loss_tot'] = loss_tot.item() / len(test_loader)
    
        if args.tqdm:
            logstr = dict2str(log_dict)
            pbar.set_description((logstr))
        
        if e % save_step == 0 and e > 0:
            ckpt_path = os.path.join(ckpt_dir, f'model-{e}.pt')
            save_ckpt(ckpt_path, model, optimizer)

    # save prediction and truth
    save_dir = os.path.join(base_dir, 'results')
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, f'results-{args.idx}.pt')

    # criterion = LpLoss()

    # model.eval()
    # with torch.no_grad():
    #     u, a_in = next(u_loader)
    #     u = u.to(device)
    #     a_in = a_in.to(device)
    #     out = model(a_in)
    #     error = criterion(out, u)
    #     print(f'Test error: {error.item()}')
    #     torch.save({'truth': u.cpu(), 'pred': out.cpu()}, result_path)
    print(f'Results saved to {result_path}')


def subprocess(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set random seed
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # create model
    model = UnetNO(args).to(device)

    num_params = count_params(model)
    print(f'Number of parameters: {num_params}')

    # Load from checkpoint
    if args.ckpt:
        ckpt_path = args.ckpt
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    # dataset
    in_data, out_data, data_mean, data_std = load_data(args.data_dir)

    # scaler = StdScaler(data_mean, data_std)    # TODO: scaler used? or not?
    
    # pack data loader
    testset = torch.utils.data.TensorDataset(in_data, out_data)
    print(len(testset))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # l2_loss_all = np.zeros((out_data.shape[0], args.repeat_run, args.sample_step))
    # residual_loss_all = np.zeros((out_data.shape[0], args.repeat_run, args.sample_step))

    optimizer = Adam(model.parameters(), lr=args.base_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.scheduler_gamma)
    
    train_ns(model, test_loader, optimizer, scheduler, device, args)
    print('Done!')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # parse options
    parser = ArgumentParser(description='Basic paser')
    # parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--idx', type=int, default=0, help='Index of the instance')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--tqdm', action='store_true', help='Turn on the tqdm')

    parser.add_argument('--re', type=float, default=1000, help='Reynold number')
    parser.add_argument('--dt', type=float, default=1/32, help='time duration at each step')
    parser.add_argument('--save_step', type=int, default=1000, help='save_step')
    parser.add_argument('--data_weight', type=float, default=0.5, help='data loss weight')
    parser.add_argument('--pde_weight', type=float, default=0.5, help='pde loss weight')

    parser.add_argument('--logdir', type=str, default='test', help='log directory')
    parser.add_argument('--num_iter', type=int, default=10001, help='number of iterations or epochs')
    parser.add_argument('--data_dir', type=str, default='data/kf_2d_re1000_256_40seed.npy', help='data directory')    
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--milestones', type=int, default=[2000, 4000, 6000, 8000], nargs='+', help='learning rate decreased')
    parser.add_argument('--base_lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='learning rate decreasing rate')

    parser.add_argument('--image_size', type=int, default=256, help='spatial resolution')
    parser.add_argument('--in_channels', type=int, default=3, help='in channel dimension')
    parser.add_argument('--out_ch', type=int, default=3, help='out channel dimension')
    parser.add_argument('--ch', type=int, default=64, help='hidden channel dimension')
    parser.add_argument('--ch_mult', type=int, default=[1, 1, 1, 2], nargs='+', help='channle multiplication')
    parser.add_argument('--num_res_blocks', type=int, default=1, help='number of residual blocks')
    parser.add_argument('--attn_resolutions', type=int, default=[16,], help='attention resolution')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--resamp_with_conv', type=bool, default=True, help='resampling with convolution')

    args = parser.parse_args()
    print(args)

    if args.seed is None:
        args.seed = random.randint(0, 100000)
    subprocess(args)
