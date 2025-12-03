import datetime
import os
import random
import numpy as np
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"]="0" # Select the available GPU
from models.twod_unet import Unet
from models.PCNO import PCNO2d

from PIL import Image
import imageio
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import colorsys

from functools import partial
from utils23 import pde_data, LpLoss, nse, corr, critical_success_index, eq_check_rt, eq_check_rf
from loss import CustomMSELoss, PearsonCorrelationScore, ScaledLpLoss
from ema import ExponentialMovingAverage
from diffusers.schedulers import DDPMScheduler
# from data.datamodule import PDEDataModule

import math
import scipy
import numpy as np
from timeit import default_timer
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import h5py
import xarray as xr
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from models.diffusion import ElucidatedDiffusion

from consistency_models import (
    ConsistencySamplingAndEditing,
    ImprovedConsistencyTraining,
    pseudo_huber_loss,
)

from consistency_models.consistency_models import pseudo_huber_loss, improved_timesteps_schedule, karras_schedule, lognormal_timestep_distribution, pad_dims_like, improved_loss_weighting
from consistency_models.utils import update_ema_model_

torch.set_num_threads(1)



################################################################
# configs
################################################################

parser = argparse.ArgumentParser()

parser.add_argument("--results_path", type=str, default="/Path_to/DiffPCNO/results/Flood/", help="path to store results")
parser.add_argument("--suffix", type=str, default="seed1", help="suffix to add to the results path")
parser.add_argument("--txt_suffix", type=str, default="Flood_DiffPCNO_seed1", help="suffix to add to the results txt")
parser.add_argument("--super", type=str, default=False, help="enable superres testing")
parser.add_argument("--verbose",type=str, default=True)
parser.add_argument("--rain",type=str, default=True)

# Pakistan flood
## Flood forecasting with T = 24h
# parser.add_argument("--T", type=int, default=287, help="number of timesteps to predict")
# parser.add_argument("--ntrain", type=int, default=10, help="training sample size")
# parser.add_argument("--nvalid", type=int, default=2, help="valid sample size")
# parser.add_argument("--ntest", type=int, default=2, help="test sample size")
## Flood forecasting with T = 12h
# parser.add_argument("--T", type=int, default=143, help="number of timesteps to predict")
# parser.add_argument("--ntrain", type=int, default=20, help="training sample size")
# parser.add_argument("--nvalid", type=int, default=4, help="valid sample size")
# parser.add_argument("--ntest", type=int, default=4, help="test sample size")

# Australia flood
## Flood forecasting with T = 24h
# parser.add_argument("--T", type=int, default=287, help="number of timesteps to predict")
# parser.add_argument("--ntrain", type=int, default=8, help="training sample size")
# parser.add_argument("--nvalid", type=int, default=1, help="valid sample size")
# parser.add_argument("--ntest", type=int, default=1, help="test sample size")
## Flood forecasting with T = 12h
parser.add_argument("--T", type=int, default=143, help="number of timesteps to predict")
parser.add_argument("--ntrain", type=int, default=16, help="training sample size")
parser.add_argument("--nvalid", type=int, default=2, help="valid sample size")
parser.add_argument("--ntest", type=int, default=2, help="test sample size")
parser.add_argument("--nsuper", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--model_type", type=str, default='Unet')
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--modes", type=int, default=12)
parser.add_argument("--width", type=int, default=20)
parser.add_argument("--Gwidth", type=int, default=10, help="hidden dimension of equivariant layers if model_type=hybrid")
parser.add_argument("--n_equiv", type=int, default=3, help="number of equivariant layers if model_type=hybrid")
parser.add_argument("--reflection", action="store_true", help="symmetry group p4->p4m for data augmentation")
parser.add_argument("--grid", type=str, default='cartesian', help="[symmetric, cartesian, None]")

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--early_stopping", type=int, default=50, help="stop if validation error does not improve for successive epochs")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--step", action="store_true", help="use step scheduler")
parser.add_argument("--gamma", type=float, default=None, help="gamma for step scheduler")
parser.add_argument("--step_size", type=int, default=None, help="step size for step scheduler")
parser.add_argument("--lmbda", type=float, default=0.0001, help="weight decay for adam")
parser.add_argument("--strategy", type=str, default="markov", help="markov, recurrent or oneshot")
parser.add_argument("--time_pad", action="store_true", help="pad the time dimension for strategy=oneshot")
parser.add_argument("--noise_std", type=float, default=0.00, help="amount of noise to inject for strategy=markov")

# Model parameters
parser.add_argument("--max_num_steps", type=int, default=32, help="Maximum number of steps")
parser.add_argument("--criterion", type=str, default="mse", help="Loss criterion")
parser.add_argument("--param_conditioning", type=str, default=None, help="Parameter conditioning")
parser.add_argument("--padding_mode", type=str, default="circular", help="Padding mode")
parser.add_argument("--predict_difference", type=bool, default=False, help="Predict difference flag")
parser.add_argument("--difference_weight", type=float, default=0.3, help="Difference weight")
parser.add_argument("--min_noise_std", type=float, default=4e-7, help="Minimum noise standard deviation")
parser.add_argument("--ema_decay", type=float, default=0.995, help="Minimum noise standard deviation")
parser.add_argument("--num_refinement_steps", type=int, default=99, help="Number of refinement steps")
parser.add_argument("--time_history", type=int, default=1, help="Time history steps")
parser.add_argument("--time_future", type=int, default=1, help="Time future steps")
parser.add_argument("--time_gap", type=int, default=0, help="Time gap")

# Data PDE parameters
parser.add_argument("--n_scalar_components", type=int, default=1, help="Number of scalar components in PDE")
parser.add_argument("--n_vector_components", type=int, default=0, help="Number of vector components in PDE")
parser.add_argument("--trajlen", type=int, default=140, help="Trajectory length")
parser.add_argument("--n_spatial_dim", type=int, default=2, help="Number of spatial dimensions")
parser.add_argument("--activation", type=str, default="gelu", help="activation")

parser.add_argument("--train_limit_trajectories", type=int, default=-1, help="Train limit trajectories")
parser.add_argument("--valid_limit_trajectories", type=int, default=-1, help="Validation limit trajectories")
parser.add_argument("--test_limit_trajectories", type=int, default=-1, help="Test limit trajectories")

args = parser.parse_args()

assert args.model_type in ["Unet"], f"Invalid model type {args.model_type}"
assert args.strategy in ["teacher_forcing", "markov", "recurrent", "oneshot"], "Invalid training strategy"

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)

data_aug = "aug" in args.model_type

# FNO data specs
# Pakistan flood
# Sy = 810
# Sx = 441

# Australia Flood
Sy = 536
Sx = 536

# UK Flood
# Sy = 85
# Sx = 137

# Mozambique Flood
# Sy = 151
# Sx = 138
S = 64
S_super = 4 * S # super spatial res
T_in = 1 # number of input times
T = args.T
T_super = 4 * T # prediction temporal super res
d = 2 # spatial res
num_channels = 3 if args.rain == True else 1
num_channels_y = 1




# adjust data specs based on model type and data path
threeD = args.model_type in ["FNO3d", "FNO3d_aug",
                             "GCNN3d_p4", "GCNN3d_p4m",
                             "GFNO3d_p4", "GFNO3d_p4m",
                             "radialNO3d_p4", "radialNO3d_p4m",
                             "PCNO3d"]
swe = False
rdb = False
ns = False
grid_type = "cartesian"
spatial_dims = range(1, d + 1)

if args.strategy == "oneshot":
    assert threeD, "oneshot strategy only for 3d models"

if threeD:
    assert args.strategy == "oneshot", "threeD models use oneshot strategy"
    # assert args.modes <= 8, "modes for 3d models should be leq 8"

ntrain = args.ntrain # 1000
nvalid = args.nvalid
ntest = args.ntest # 200

time_modes = None
time = args.strategy == "oneshot" # perform convolutions in space-time
if time and not args.time_pad:
    time_modes = 5 if swe else 6 # 6 is based on T=10
elif time and swe:
    time_modes = 8

modes = args.modes
width = args.width
n_layer = args.depth
batch_size = args.batch_size

epochs = args.epochs # 500
learning_rate = args.learning_rate
scheduler_step = args.step_size
scheduler_gamma = args.gamma # for step scheduler

initial_step = 1 if args.strategy == "markov" else T_in


root = '/pre_trained/Models/2D_Flood/DiffPCNO' # the path for the pre-trained checkpoints of DiffPCNO
path_model = os.path.join(root, 'model.pt')
writer = SummaryWriter(root)


################################################################
# Model init
################################################################
if args.model_type in ["Unet"]:
    model = Unet(n_input_scalar_components=args.n_scalar_components, n_input_vector_components=args.n_vector_components,
         n_output_scalar_components=args.n_scalar_components,
         n_output_vector_components=args.n_vector_components, time_history=args.time_history + args.time_future,
         time_future=args.time_future, hidden_channels=64, activation=args.activation,
         norm=True, param_conditioning=args.param_conditioning).cuda()
else:
    raise NotImplementedError("Model not recognized")

model.load_state_dict(torch.load(path_model))

# Define \theta_{-}, which is EMA of the params
ema_model = Unet(n_input_scalar_components=args.n_scalar_components, n_input_vector_components=args.n_vector_components,
         n_output_scalar_components=args.n_scalar_components,
         n_output_vector_components=args.n_vector_components, time_history=args.time_history + args.time_future,
         time_future=args.time_future, hidden_channels=64, activation=args.activation,
         norm=True, param_conditioning=args.param_conditioning).cuda()
ema_model.load_state_dict(model.state_dict())

model_con = PCNO2d(num_channels=num_channels, initial_step=initial_step, modes1=modes, modes2=modes, width=width,
                  grid_type=grid_type).cuda()
path_model_con = '/pre_trained/Models/2D_Flood/PCNO/model.pt' # the path for the pre-trained checkpoints of PCNO
model_con.load_state_dict(torch.load(path_model_con))
model_con.eval()
################################################################
# load data
################################################################
full_data = None # for superres
path_train = '/Path_to/Datasets/Australia/144/train' # training dataset
# path_val = '/mnt/HDD2/qingsong/qinqsong/Datasets/Pakistan/144/valid'
# path_test = '/mnt/HDD2/qingsong/qinqsong/Datasets/Pakistan/144/test'
path_test = '/Path_to/Datasets/Australia/1day' # test dataset for Pakistan flood, Australia flood, UK flood, or Mozambique flood

paths = [path_test]

global_max = None
global_min = None
for path in paths:
    for path_root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.pt'):
                file_path = os.path.join(path_root, file)
                try:
                    data = torch.load(file_path)
                    data = torch.log1p(data)
                    local_max = data[..., 0:1].max().item()
                    local_min = data[..., 0:1].min().item()
                    if global_max is None or local_max > global_max:
                        global_max = local_max
                    if global_min is None or local_min < global_min:
                        global_min = local_min
                except Exception as e:
                    print(f"Skipping file {file_path}, error: {e}")

print('global_max', global_max)
print('global_min', global_min)
#
# train_data = pde_data(path_train, strategy=args.strategy, T_in=T_in, T_out=T, rain=args.rain, std=args.noise_std, global_max=global_max)
# ntrain = len(train_data)

T_test = 287 # Test time step (T_test = 287 for 24h; T_test = 575 for 48h)
# path_test1 = '/mnt/HDD2/qingsong/qinqsong/Datasets/Pakistan/test/2day'
test_data = pde_data(path_test, train=False, strategy=args.strategy, T_in=T_in, T_out=T_test, rain=args.rain, global_max=global_max)
train_data = pde_data(path_train, strategy=args.strategy, T_in=T_in, T_out=T, rain=args.rain, std=args.noise_std, global_max=global_max)

# test_rt_data = pde_data(test_rt, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
# test_rf_data = pde_data(test_rf, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
ntest = len(test_data)
print('ntest', ntest)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################


y_r_max = float('-inf')
y_r_min = float('inf')
for xx, yy in train_loader:
    x = xx.cuda()
    y = yy.cuda()
    # print('y', y.shape)
    # x0 = x * (data_max - data_min) + data_min
    # y0 = y * (data_max - data_min) + data_min
    x0 = x[..., 0:1] * global_max
    x0 = torch.expm1(x0)
    x[..., 0:1] = x0
    with torch.no_grad():
        im = model_con(x)
        im_r = torch.log1p(im) / global_max
        # im_r = (im_r - 0.5) / 0.5
        # print('im_r',im_r.shape)
    y_r = y - im_r.squeeze(-2)
    y_r_max = max(y_r_max, y_r.max().item())
    y_r_min = min(y_r_min, y_r.min().item())
print('y_r_max', y_r_max)
print('y_r_min', y_r_min)
# y_r_max = 0.03968238830566406
# y_r_min = -0.011303037405014038

# model = ElucidatedDiffusion(net, channels = num_channels, image_size=S, sigma_data=sigma_data).cuda()
# scaler = GradScaler()

complex_ct = sum(par.numel() * (1 + par.is_complex()) for par in model.parameters())
real_ct = sum(par.numel() for par in model.parameters())
if args.verbose:
    print(f"{args.model_type}; # Params: complex count {complex_ct}, real count: {real_ct}")
writer.add_scalar("Parameters/Complex", complex_ct)
writer.add_scalar("Parameters/Real", real_ct)

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.lmbda)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.995))

lploss = LpLoss(size_average=False)

best_valid = float("inf")


if (args.n_spatial_dim) == 3:
    _mode = "3D"
    nn.Conv3d = partial(nn.Conv3d, padding_mode=args.padding_mode)
elif (args.n_spatial_dim) == 2:
    _mode = "2D"
    nn.Conv2d = partial(nn.Conv2d, padding_mode=args.padding_mode)
elif (args.n_spatial_dim) == 1:
    _mode = "1D"
    nn.Conv1d = partial(nn.Conv1d, padding_mode=args.padding_mode)
else:
    raise NotImplementedError(f"{pde}")
train_criterion = CustomMSELoss()
# ema = ExponentialMovingAverage(model, decay=args.ema_decay)


# implement the denoising manually.
betas = [args.min_noise_std ** (k / args.num_refinement_steps) for k in reversed(range(args.num_refinement_steps + 1))]
scheduler_DDPM = DDPMScheduler(
    num_train_timesteps=args.num_refinement_steps + 1,
    trained_betas=betas,
    prediction_type="v_prediction",
    clip_sample=False,
)
# Multiplies k before passing to frequency embedding.
time_multiplier = 1000 / args.num_refinement_steps

val_criterions = {"mse": CustomMSELoss(), "scaledl2": ScaledLpLoss()}
rollout_criterions = {"mse": torch.nn.MSELoss(reduction="none"), "corr": PearsonCorrelationScore()}
time_resolution = args.trajlen
# Max number of previous points solver can eat
reduced_time_resolution = time_resolution - args.time_history
        # Number of future points to predict
max_start_time = (reduced_time_resolution - args.time_future * args.max_num_steps - args.time_gap)
max_start_time = max(0, max_start_time)

def predict_next_solution(model, x):
    # m = x[..., :2] + torch.randn_like(x[..., 2:]).cuda() * 80.0
    # print('m',m.shape)
    # print('x',x.shape)
    # x_c = x[..., :2]
    # x_c = (x_c - data_min) / (data_max - data_min)
    # x_c = (x_c - 0.5) / 0.5
    y = model.sample(
        torch.randn_like(x[..., 1:2]).cuda() * 80.0,
        list(reversed([0.661, 0.9, 5.84, 24.4, 80.0])),
        x,
    )
    # if args.predict_difference:
    #     y = y * args.difference_weight + x[:,1:,:,:]

    y = y.clamp(-1, 1)
    y = y * 0.5 + 0.5
    y = y * (y_r_max - y_r_min) + y_r_min
    return y

def get_eval_pred_m(model, model_con, x, y_c, y_g, strategy, T, times, num_samples=10):
    all_preds = []
    all_preds_r = []
    mean_pred = None
    mean_pred_r = None
    n = 0
    for i in range(num_samples):
        pred = None
        pred_r = None
        x0 = x
        for t in range(T):
            t1 = default_timer()
            # x0 = x * 0.5 + 0.5
            with torch.no_grad():
                im_c = model_con(x0)
                im_cr = torch.log1p(im_c) / global_max
                # im_cr = (im_cr - 0.5) / 0.5
            im_cc = torch.cat((im_c, x0), dim=-1)
            im_r = predict_next_solution(model, im_cc)
            im = im_cr + im_r
            # im = im.unsqueeze(-2)
            # print('im',im.shape)
            times.append(default_timer() - t1)
            if t == 0:
                pred = im
                # pred_r = im_r
            else:
                pred = torch.cat((pred, im), -2)
                # pred_r = torch.cat((pred_r, im_r), -2)

            if strategy == "markov":
                # x = im_c
                # x = x.clamp(0, 1)
                y_cc = y_c[..., t:(t + 1), :]
                # im_g = im * global_max
                # im_g = torch.expm1(im_g)
                # im_g = torch.clamp(im_g, min=0)
                x0 = torch.cat((im_c, y_cc), dim=-1)
            else:
                x0 = torch.cat((x0[..., 1:, :], im), dim=-2)
        pred_new = pred * global_max
        pred_new = torch.expm1(pred_new)
        pred_new = torch.clamp(pred_new, min=0)
        error_f = (pred_new - y_g) ** 2
        n += 1
        if mean_pred is None:
            mean_pred = pred.clone()
            M2 = torch.zeros_like(pred)
        else:
            delta = pred - mean_pred
            mean_pred += delta / n
            M2 += delta * (pred - mean_pred)

        if mean_pred_r is None:
            mean_pred_r = error_f.clone()
            M2_r = torch.zeros_like(error_f)
        else:
            delta_r = error_f - mean_pred_r
            mean_pred_r += delta_r / n
            M2_r += delta_r * (error_f - mean_pred_r)
    std_pred = torch.sqrt(M2 / (n - 1))
    std_pred_r = torch.sqrt(M2_r / (n - 1))
    print('mean_pred', mean_pred.shape)
    print('std_pred', std_pred.shape)
    print('mean_pred_r', mean_pred_r.shape)
    print('std_pred_r', std_pred_r.shape)
    return mean_pred, std_pred, mean_pred_r, std_pred_r
    #     all_preds.append(pred.unsqueeze(0).cpu())
    #     # all_preds_r.append(pred_r.unsqueeze(0))
    # all_preds = torch.cat(all_preds, dim=0)
    # # all_preds_r = torch.cat(all_preds_r, dim=0)
    #
    # mean_pred = all_preds.mean(dim=0)
    # std_pred = all_preds.std(dim=0)
    # print('mean_pred', mean_pred.shape)
    # print('std_pred', std_pred.shape)
    # # std_pred_r = all_preds_r.std(dim=0)
    #
    # return mean_pred, std_pred

def apply_ema():
    ema.apply_shadow()
def remove_ema():
    ema.restore()

model.eval()

start = default_timer()
if args.verbose:
    print("Training...")
step_ct = 0
train_times = []
eval_times = []
# ema.register()
end_scales = 150
start_scales = 2
sigma_max = 80.0
rho =7.0
sigma_min = 0.002
consistency_training=ImprovedConsistencyTraining(final_timesteps=11)
initial_timesteps = 10
final_timesteps = 1280
lognormal_mean = -1.1
lognormal_std = 2.0
# consistency_sampling=ConsistencySamplingAndEditing()
sigma_data = 0.5
sigma_min = 0.002
ema_decay_rate = 0.99993
for param in model_con.parameters():
    param.requires_grad = False

current_training_step = 0
# total_training_steps = num_training_steps
def output_scale(x, x_out, sigma, sigma_data, sigma_min):
    c_skip = sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)
    c_out = (sigma_data * (sigma - sigma_min)) / (sigma_data**2 + sigma**2) ** 0.5
    c_skip = pad_dims_like(c_skip, x)
    c_out = pad_dims_like(c_out, x)
    return c_skip * x + c_out * x_out
def generate_movie_2D(key, test_x, test_y, preds_y, preds_std, plot_title='', field=0, val_cbar_index=-1, err_cbar_index=-1,
                      val_clim=None, err_clim=None, std_clim=None, font_size=None, movie_dir='', movie_name='movie.gif',
                      frame_basename='movie', frame_ext='jpg', remove_frames=True):
    frame_files = []

    if movie_dir:
        os.makedirs(movie_dir, exist_ok=True)

    if font_size is not None:
        plt.rcParams.update({'font.size': font_size})

    if len(preds_y.shape) == 4:
        Nsamples, Nx, Ny, Nt = preds_y.shape
        preds_y = preds_y.reshape(Nsamples, Nx, Ny, Nt, 1)
        test_y = test_y.reshape(Nsamples, Nx, Ny, Nt, 1)
        preds_std = preds_std.reshape(Nsamples, Nx, Ny, Nt, 1)
    Nsamples, Nx, Ny, Nt, Nfields = preds_y.shape
    # print('preds_y', preds_y.shape)

    pred = preds_y[key, ..., field]
    true = test_y[key, ..., field]
    std = preds_std[key, ..., field]
    error = torch.abs(pred - true)

    a = test_x[key]
    x = torch.linspace(0, 1, Nx + 1)[:-1]
    y = torch.linspace(0, 1, Ny + 1)[:-1]
    X, Y = torch.meshgrid(x, y)
    # print('X', X.shape)
    # print('Y', Y.shape)

    fig, axs = plt.subplots(1, 4, figsize=(30, 4))
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]
    ax4 = axs[3]
    colors = plt.cm.viridis(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]
    # cmap = ListedColormap(colors)

    pcm1 = ax1.pcolormesh(X, Y, true[..., val_cbar_index], cmap='RdBu_r', label='true', shading='gouraud')
    pcm2 = ax2.pcolormesh(X, Y, pred[..., val_cbar_index], cmap='RdBu_r', label='pred', shading='gouraud')
    pcm3 = ax3.pcolormesh(X, Y, error[..., err_cbar_index], cmap='RdBu_r', label='error', shading='gouraud')
    pcm4 = ax4.pcolormesh(X, Y, std[..., err_cbar_index], cmap='RdBu_r', label='uncertainty', shading='gouraud')

    if val_clim is None:
        val_clim = pcm1.get_clim()
    if err_clim is None:
        err_clim = pcm3.get_clim()
    if std_clim is None:
        std_clim = pcm4.get_clim()

    pcm1.set_clim(val_clim)
    plt.colorbar(pcm1, ax=ax1)
    ax1.set_aspect('auto')
    # ax1.axis('square')

    pcm2.set_clim(val_clim)
    plt.colorbar(pcm2, ax=ax2)
    ax2.set_aspect('auto')
    # ax2.axis('square')

    pcm3.set_clim(err_clim)
    plt.colorbar(pcm3, ax=ax3)
    ax3.set_aspect('auto')
    # ax3.axis('square')

    pcm4.set_clim(std_clim)
    plt.colorbar(pcm4, ax=ax4)
    ax4.set_aspect('auto')

    plt.tight_layout()

    for i in range(Nt):
        # Exact
        ax1.clear()
        pcm1 = ax1.pcolormesh(X, Y, true[..., i], cmap='RdBu_r', label='true', shading='gouraud')
        pcm1.set_clim(val_clim)
        ax1.set_aspect('auto')
        ax1.set_xlim(X.min(), X.max())
        ax1.set_ylim(Y.min(), Y.max())
        ax1.set_axis_off()
        # ax1.set_title(f'Ground truth {plot_title}')
        # ax1.axis('square')

        # Predictions
        ax2.clear()
        pcm2 = ax2.pcolormesh(X, Y, pred[..., i], cmap='RdBu_r', label='pred', shading='gouraud')
        pcm2.set_clim(val_clim)
        # ax2.set_title(f'Case 1 (DNO-3) {plot_title}')
        ax2.set_aspect('auto')
        ax2.set_xlim(X.min(), X.max())
        ax2.set_ylim(Y.min(), Y.max())
        ax2.set_axis_off()
        # ax2.axis('square')

        # Error
        ax3.clear()
        pcm3 = ax3.pcolormesh(X, Y, error[..., i], cmap='RdBu_r', label='error', shading='gouraud')
        pcm3.set_clim(err_clim)
        # ax3.set_title(f'Errors {plot_title}')
        ax3.set_aspect('auto')
        ax3.set_xlim(X.min(), X.max())
        ax3.set_ylim(Y.min(), Y.max())
        ax3.set_axis_off()
        # ax3.axis('square')

        # Uncertainty
        ax4.clear()
        pcm4 = ax4.pcolormesh(X, Y, std[..., i], cmap='RdBu_r', label='uncertainty', shading='gouraud')
        pcm4.set_clim(std_clim)
        ax4.set_aspect('auto')
        ax4.set_xlim(X.min(), X.max())
        ax4.set_ylim(Y.min(), Y.max())
        ax4.set_axis_off()

        #         plt.tight_layout()
        # fig.canvas.draw()
        #
        if movie_dir:
            frame_path = os.path.join(movie_dir, f'{frame_basename}-{i:03}.{frame_ext}')
            frame_files.append(frame_path)
            plt.savefig(frame_path, dpi=900, bbox_inches='tight')
        # plt.draw()
        # plt.pause(0.1)
    #
    # if movie_dir:
    #     movie_path = os.path.join(movie_dir, movie_name)
    #     with imageio.get_writer(movie_path, mode='I') as writer:
    #         for frame in frame_files:
    #             image = imageio.imread(frame)
    #             writer.append_data(image)
    # plt.show()

    # if movie_dir and remove_frames:
    #     for frame in frame_files:
    #         try:
    #             os.remove(frame)
    #         except:
    #             pass

# test
model.load_state_dict(torch.load(path_model))
model.eval()
test_vort_l2 = test_pres_l2 = 0
test_l2 = test_nse = test_corr = test_csi_1 = test_csi_2 = test_csi_3 = test_csi_4 = 0
rotations_l2 = 0
reflections_l2 = 0
test_rt_l2 = 0
test_rf_l2 = 0
test_loss_by_channel = None
key = 0
i = 0
log_dir = '/Path_to/Flood/Results/Australia/144_288_d'
with torch.no_grad():
    # apply_ema()
    for xx, yy in test_loader:
        x = xx.cuda()
        yy = yy.cuda()
        # x = (xx * 0.5 + 0.5)
        input_data = x
        yy_g = yy[..., 0:1]
        yy_c = yy[..., 1:3]

        x0 = x[..., 0:1] * global_max
        x0 = torch.expm1(x0)
        x[..., 0:1] = x0
        yy_g = yy_g * global_max
        yy_g = torch.expm1(yy_g)
        # x = x * 0.5 + 0.5
        # y = y * 0.5 + 0.5
        # cond = cond.cuda()
        # im = model_con(x, z=cond)
        # im_c = torch.cat((im, x), dim=1)

        pred, pred_std, pred_r, pred_std_r = get_eval_pred_m(model=model, model_con=model_con, x=x, y_c=yy_c, y_g=yy_g, strategy=args.strategy, T=T_test, times=[], num_samples=50)
        pred = pred * global_max
        pred = torch.expm1(pred)
        pred = torch.clamp(pred, min=0)
        print('pred', pred.shape)
        print('pred_std', pred_std.shape)
        print('pred_r', pred_r.shape)
        print('pred_std_r', pred_std_r.shape)
        test_l2 += lploss(pred.reshape(len(pred), -1, num_channels_y), yy_g.reshape(len(yy_g), -1, num_channels_y)).item()
        test_nse += nse(pred.reshape(len(pred), -1, num_channels_y), yy_g.reshape(len(yy_g), -1, num_channels_y)).item()
        test_corr += corr(pred.reshape(len(pred), -1, num_channels_y),
                          yy_g.reshape(len(yy_g), -1, num_channels_y)).item()
        test_csi_1 += critical_success_index(pred.reshape(len(pred), -1, 1),
                                             yy_g.reshape(len(yy_g), -1, 1), 0.01).item()
        test_csi_2 += critical_success_index(pred.reshape(len(pred), -1, 1),
                                             yy_g.reshape(len(yy_g), -1, 1), 0.05).item()
        test_csi_3 += critical_success_index(pred.reshape(len(pred), -1, 1),
                                             yy_g.reshape(len(yy_g), -1, 1), 0.1).item()
        test_csi_4 += critical_success_index(pred.reshape(len(pred), -1, 1),
                                             yy_g.reshape(len(yy_g), -1, 1), 0.5).item()

        y_hm = yy_g
        pred_hm = pred
        pred_std_hm = pred_std
        pred_r_hm = pred_r
        pred_std_r_hm = pred_std_r

        gt_f = y_hm[..., 0]
        pred_f = pred_hm[..., 0]
        pred_std_f = pred_std_hm[..., 0]
        r_f = pred_r_hm[..., 0]
        std_r_f = pred_std_r_hm[..., 0]
        hg01, hg05 = F.threshold(gt_f, threshold=0.1, value=0), F.threshold(gt_f, threshold=0.5, value=0)
        hp01, hp05 = F.threshold(pred_f, threshold=0.1, value=0), F.threshold(pred_f, threshold=0.5, value=0)
        hg_a, hg01_a, hg05_a = gt_f[0, ..., -1], hg01[0, ..., -1], hg05[0, ..., -1]
        hg_a, hg01_a, hg05_a = hg_a.cpu().detach().numpy(), hg01_a.cpu().detach().numpy(), hg05_a.cpu().detach().numpy()
        hp_a, hp01_a, hp05_a = pred_f[0, ..., -1], hp01[0, ..., -1], hp05[0, ..., -1]
        hp_a, hp01_a, hp05_a = hp_a.cpu().detach().numpy(), hp01_a.cpu().detach().numpy(), hp05_a.cpu().detach().numpy()
        print('pred_std_f', pred_std_f.shape)
        pred_std_a = pred_std_f[0, ..., -1]
        pred_std_a = pred_std_a.cpu().detach().numpy()
        print('pred_std_a', pred_std_a.shape)
        time_n = (i+1) * T_test
        os.makedirs(os.path.join(log_dir, 'hg'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'hg01'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'hg05'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'hp'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'hp01'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'hp05'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'pred_std'), exist_ok=True)

        hg_aa = Image.fromarray(hg_a)
        hg01_aa = Image.fromarray(hg01_a)
        hg05_aa = Image.fromarray(hg05_a)
        hp_aa = Image.fromarray(hp_a)
        hp01_aa = Image.fromarray(hp01_a)
        hp05_aa = Image.fromarray(hp05_a)
        pred_std_aa = Image.fromarray(pred_std_a)

        hg_aa.save(os.path.join(log_dir, 'hg/%s.tiff' % time_n))
        hg01_aa.save(os.path.join(log_dir, 'hg01/%s.tiff' % time_n))
        hg05_aa.save(os.path.join(log_dir, 'hg05/%s.tiff' % time_n))
        hp_aa.save(os.path.join(log_dir, 'hp/%s.tiff' % time_n))
        hp01_aa.save(os.path.join(log_dir, 'hp01/%s.tiff' % time_n))
        hp05_aa.save(os.path.join(log_dir, 'hp05/%s.tiff' % time_n))
        pred_std_aa.save(os.path.join(log_dir, 'pred_std/%s.tiff' % time_n))


        if i == 0:
            g_f = gt_f
        else:
            g_f = torch.cat((g_f, gt_f), 0)
        print('g_f', g_f.shape)

        if i == 0:
            p_f = pred_f
        else:
            p_f = torch.cat((p_f, pred_f), 0)

        if i == 0:
            p_std_f = pred_std_f
        else:
            p_std_f = torch.cat((p_std_f, pred_std_f), 0)

        if i == 0:
            p_r_f = r_f
        else:
            p_r_f = torch.cat((p_r_f, r_f), 0)

        if i == 0:
            p_r_std_f = std_r_f
        else:
            p_r_std_f = torch.cat((p_r_std_f, std_r_f), 0)

        # Visulazation
        gt_u = yy_g[..., -1, :]
        pred_u = pred[..., -1, :]
        pred_stdu = pred_std[..., -1, :]
        # U
        gt_um = torch.rot90(gt_u, k=-1, dims=[1, 2])
        print('gt_um', gt_um.shape)

        # pred_umm = pred_u.permute(0, 3, 1, 2)
        out_um = torch.rot90(pred_u, k=-1, dims=[1, 2])

        # pred_std_umm = pred_stdu.permute(0, 3, 1, 2)
        outm_ustd = torch.rot90(pred_stdu, k=-1, dims=[1, 2])

        print('pred_u', out_um.shape)
        print('outm_ustd', outm_ustd.shape)
        # MOIVE
        movie_dir = os.path.join(log_dir, 'pred/%s/' % (str(i)))
        os.makedirs(movie_dir, exist_ok=True)
        movie_name = 'U.gif'
        frame_basename = 'U_frame'
        frame_ext = 'jpg'
        plot_title = ""
        field = 0
        val_cbar_index = -1
        err_cbar_index = -1
        font_size = 12
        remove_frames = True
        generate_movie_2D(key, input_data.cpu(), gt_um.cpu(), out_um.cpu(), outm_ustd.cpu(),
                          plot_title=plot_title,
                          field=field,
                          val_cbar_index=val_cbar_index,
                          err_cbar_index=err_cbar_index,
                          movie_dir=movie_dir,
                          movie_name=movie_name,
                          frame_basename=frame_basename,
                          frame_ext=frame_ext,
                          remove_frames=remove_frames,
                          font_size=font_size)

        i = i + 1
    # remove_ema()
    # writer.add_scalar("Test/Loss", test_l2 / ntest, best_epoch)

test_time_l2 = test_space_l2 = ntest_super = test_int_space_l2 = test_int_time_l2 = None

# gt_f = g_f.cpu().numpy()
# pred_f = p_f.cpu().numpy()
# pred_std_f = p_std_f.cpu().numpy()
# e_f = p_r_f.cpu().numpy()
# e_std_f = p_r_std_f.cpu().numpy()
# g_avg = np.mean(gt_f, axis=(1, 2))
# p_avg = np.mean(pred_f, axis=(1, 2))
# p_std_avg = np.mean(pred_std_f, axis=(1, 2))
# e_avg = np.mean(e_f, axis=(1, 2))
# e_std_avg = np.mean(e_std_f, axis=(1, 2))
# # file1 = os.path.join(log_dir, 'gt.xlsx')
# # file2 = os.path.join(log_dir, 'pre.xlsx')
# file1 = os.path.join(log_dir, 'gt_ave.xlsx')
# file2 = os.path.join(log_dir, 'pre_ave.xlsx')
# file3 = os.path.join(log_dir, 'e_ave.xlsx')
# file4 = os.path.join(log_dir, 'e_std_ave.xlsx')
# # file5 = os.path.join(log_dir, 'pre_std.xlsx')
# file5 = os.path.join(log_dir, 'pre_std_ave.xlsx')
# #
# # # #
# g_mean = g_avg.reshape(-1, 1)
# p_mean = p_avg.reshape(-1, 1)
# p_std_mean = p_std_avg.reshape(-1, 1)
# e_mean = e_avg.reshape(-1, 1)
# e_std_mean = e_std_avg.reshape(-1, 1)
# print('e_mean', e_mean.shape)
# print('e_std_mean', e_std_mean.shape)
# #
# # # #
# # df_g = pd.DataFrame(g_avg.T)
# # df_g.to_excel(file1, index=False, header=False)
# # df_p = pd.DataFrame(p_avg.T)
# # df_p.to_excel(file2, index=False, header=False)
# # df_p_std = pd.DataFrame(p_std_avg.T)
# # df_p_std.to_excel(file6, index=False, header=False)
# # #
# # #
# df_gs = pd.DataFrame(g_mean, columns=['DiffPCNO'])
# df_gs.to_excel(file1, index=False)
# df_ps = pd.DataFrame(p_mean, columns=['DiffPCNO'])
# df_ps.to_excel(file2, index=False)
# df_es = pd.DataFrame(e_mean, columns=['DiffPCNO'])
# df_es.to_excel(file3, index=False)
# df_ep = pd.DataFrame(e_std_mean, columns=['DiffPCNO'])
# df_ep.to_excel(file4, index=False)
# df_pss = pd.DataFrame(p_std_mean, columns=['DiffPCNO'])
# df_pss.to_excel(file5, index=False)


print(f"{args.model_type} done training for Target domain; \nTest: {test_l2 / ntest}, Test_nse: {test_nse / ntest}, Test_corr: {test_corr / ntest}, Test_csi_1: {test_csi_1 / ntest}, "
      f"Test_csi_2: {test_csi_2 / ntest}, Test_csi_3: {test_csi_3 / ntest}, Test_csi_4: {test_csi_4 / ntest}")

writer.flush()
writer.close()