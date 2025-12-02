"""
This is a modified version of fourier_2d_time.py from https://github.com/zongyi-li/fourier_neural_operator
"""
import datetime
import os
import random
import numpy as np
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"]="0" # Select the available GPU
from models.oned_unet import Unet
from models.PCNO1D import PCNO1d
# from models.GFNO_steerable import GFNO2d_steer
# from models.Unet import Unet_Rot, Unet_Rot_M, Unet_Rot_3D
# from models.Ghybrid import Ghybrid2d
# from models.radialNO import radialNO2d, radialNO3d
# from models.GCNN import GCNN2d, GCNN3d
from PIL import Image
import imageio
import matplotlib.pyplot as plt

from functools import partial
from KS.utils_ks import pde_data, LpLoss, eq_check_rt, eq_check_rf, kerras_boundaries # utils_ks is for 1D KSE with varying viscosity
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

parser.add_argument("--results_path", type=str, default="/Path_to/results/KDE/DiffPCNO/", help="path to store results")
parser.add_argument("--suffix", type=str, default="seed1", help="suffix to add to the results path")
parser.add_argument("--txt_suffix", type=str, default="1D_KDE_DiffPCNO_2_seed1", help="suffix to add to the results txt")
parser.add_argument("--super", type=str, default='False', help="enable superres testing")
parser.add_argument("--verbose",type=str, default='True')

parser.add_argument("--T", type=int, default=20, help="number of timesteps to predict")
parser.add_argument("--ntrain", type=int, default=1000, help="training sample size")
parser.add_argument("--nvalid", type=int, default=100, help="valid sample size")
parser.add_argument("--ntest", type=int, default=100, help="test sample size")
parser.add_argument("--nsuper", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--model_type", type=str, default='Unet')
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--modes", type=int, default=12)
parser.add_argument("--width", type=int, default=20)
parser.add_argument("--Gwidth", type=int, default=10, help="hidden dimension of equivariant layers if model_type=hybrid")
parser.add_argument("--n_equiv", type=int, default=3, help="number of equivariant layers if model_type=hybrid")
parser.add_argument("--reflection", action="store_true", help="symmetry group p4->p4m for data augmentation")
parser.add_argument("--grid", type=str, default=None, help="[symmetric, cartesian, None]")

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--early_stopping", type=int, default=100, help="stop if validation error does not improve for successive epochs")
parser.add_argument("--batch_size", type=int, default=128)
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
parser.add_argument("--param_conditioning", type=str, default="scalar_3", help="Parameter conditioning for 1D KSE with varying viscosity")
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
parser.add_argument("--n_spatial_dim", type=int, default=1, help="Number of spatial dimensions")
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
S = 256 # spatial res
S_super = 4 * S # super spatial res
T_in = 1 # number of input times
T_train = 139
T_valid = 639
T_test = 639
d = 2 # spatial res
num_channels = 1



# adjust data specs based on model type and data path
threeD = args.model_type in ["FNO3d", "FNO3d_aug",
                             "GCNN3d_p4", "GCNN3d_p4m",
                             "GFNO3d_p4", "GFNO3d_p4m",
                             "radialNO3d_p4", "radialNO3d_p4m",
                             "PCNO3d"]

grid_type = "symmetric"
swe = False
if args.grid:
    grid_type = args.grid
    assert grid_type in ['symmetric', 'cartesian', 'None']


spatial_dims = range(1, d + 1)

if args.strategy == "oneshot":
    assert threeD, "oneshot strategy only for 3d models"

if threeD:
    assert args.strategy == "oneshot", "threeD models use oneshot strategy"
    # assert args.modes <= 8, "modes for 3d models should be leq 8"

ntrain = args.ntrain # 1000
nvalid = args.nvalid
ntest = args.ntest # 100

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

root = args.results_path + f"/{'_'.join(str(datetime.datetime.now()).split())}"
if args.suffix:
    root += "_" + args.suffix
os.makedirs(root)
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
         norm=True, param_conditioning=args.param_conditioning, n_dims=args.n_spatial_dim).cuda()
else:
    raise NotImplementedError("Model not recognized")

# Define \theta_{-}, which is EMA of the params
ema_model = Unet(n_input_scalar_components=args.n_scalar_components, n_input_vector_components=args.n_vector_components,
         n_output_scalar_components=args.n_scalar_components,
         n_output_vector_components=args.n_vector_components, time_history=args.time_history + args.time_future,
         time_future=args.time_future, hidden_channels=64, activation=args.activation,
         norm=True, param_conditioning=args.param_conditioning, n_dims=args.n_spatial_dim).cuda()
ema_model.load_state_dict(model.state_dict())

model_con = PCNO1d(num_channels=num_channels, hidden_channels=64, param_conditioning=args.param_conditioning, modes=modes, width=width, initial_step=initial_step).cuda()
path_model_con = 'Path_to/Models/1D_KSE/PCNO/Varying_viscosity/model.pt'
model_con.load_state_dict(torch.load(path_model_con))
model_con.eval()
################################################################
# load data
################################################################
def dataset(path, mode):
    with h5py.File(path, "r") as f:
        data_h5 = f[mode]
        data_key = [k for k in data_h5.keys() if k.startswith("pde_")][0]
        data = {
            "u": torch.tensor(data_h5[data_key][:].astype(np.float32)),
            "dt": torch.tensor(data_h5["dt"][:].astype(np.float32)),
            "dx": torch.tensor(data_h5["dx"][:].astype(np.float32)),
        }
        # data_u = data["u"].permute(0, 2, 1)

        # print('data["u"]', data["u"].shape)
        # print('data["dt"]', data["dt"].shape)
        # print('data["dx"]', data["dx"].shape)
        if "v" in data_h5:
            data["v"] = torch.tensor(data_h5["v"][:].astype(np.float32))

        data["orig_dt"] = data["dt"].clone()
        if data["u"].ndim == 3:
            data["u"] = data["u"].unsqueeze(dim=-2)  # Add channel dimension

        # Normalizing the data to [0,1]

        shift_u = data["u"].min()
        scale_u = data["u"].max() - data["u"].min()
        data["u"] = (data["u"] - shift_u) / scale_u
        
        # The KS equation is parameterized by [1] the time step between observations
        # (measured in seconds, usually around 0.2), [2] the spatial step between
        # data points in the spatial domain (measured in meters, usually around 0.2),
        # and finally [3] the viscosity parameter (measured in m^2/s, usually between 0.5 - 1.5).
        # We scale these parameters to be in the range [0, 10] to be visible changes in fourier embeds.
        # This accelerates learning and makes it easier for the models to learn the conditional dynamics.
        # # Scaling time step.
        if data["dt"].min() > 0.15 and data["dt"].max() < 0.25:
            data["dt"] = (data["dt"] - 0.15) * 100.0
        else:
            print(
                f"WARNING: dt is not in the expected range (min {data['dt'].min()}, max {data['dt'].max()}, mean {data['dt'].mean()}) - scaling may be incorrect."
            )
        # Scaling spatial step.
        if data["dx"].min() > 0.2 and data["dx"].max() < 0.3:
            data["dx"] = (data["dx"] - 0.2) * 100.0
        else:
            print(
                f"WARNING: dx is not in the expected range (min {data['dx'].min()}, max {data['dx'].max()}, mean {data['dx'].mean()}) - scaling may be incorrect."
            )
        # Scaling viscosity.
        if "v" in data:
            if data["v"].min() >= 0.5 and data["v"].max() <= 1.5:
                data["v"] = (data["v"] - 0.5) * 100.0
            else:
                print(
                    f"WARNING: v is not in the expected range (min {data['v'].min()}, max {data['v'].max()}, mean {data['v'].mean()}) - scaling may be incorrect."
                )
        # print('data["u"]', data["u"].shape)
        # print('data["dt"]', data["dt"].shape)
        # print('data["dx"]', data["dx"].shape)
    return data

Path_train = '/Path_to/Dataset/KS/KS_train_conditional_viscosity.h5'
Path_valid = '/Path_to/Dataset/KS/KS_valid_conditional_viscosity.h5'
Path_test = '/Path_to/Dataset/KS/KS_test_conditional_viscosity.h5'

train = dataset(path=Path_train, mode='train')
test = dataset(path=Path_test, mode='test')
valid = dataset(path=Path_valid, mode='valid')


train_data = pde_data(train, train=True, strategy=args.strategy, T_in=T_in, T_out=T_train, std=args.noise_std)
ntrain = len(train_data)
valid_data = pde_data(valid, train=False, strategy=args.strategy, T_in=T_in, T_out=T_valid)
nvalid = len(valid_data)
test_data = pde_data(test, train=False, strategy=args.strategy, T_in=T_in, T_out=T_test)
ntest = len(test_data)
print('ntrain',ntrain)
print('nvalid',nvalid)
print('ntest',ntest)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################


y_r_max = float('-inf')
y_r_min = float('inf')
for xx, yy, cond in train_loader:
    x = xx.cuda()
    y = yy.cuda()
    cond = cond.cuda()
    # x0 = x * (data_max - data_min) + data_min
    # y0 = y * (data_max - data_min) + data_min
    # x0 = x * (data_max - data_min) + data_min
    with torch.no_grad():
        im = model_con(x, z=cond)
        # im = (im - 0.5) / 0.5
    im_c = torch.cat((im, x), dim=1)
    y_r = y - im
    y_r_max = max(y_r_max, y_r.max().item())
    y_r_min = min(y_r_min, y_r.min().item())
print('y_r_max', y_r_max)
print('y_r_min', y_r_min)

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
if args.step:
    assert args.step_size is not None, "step_size is None"
    assert scheduler_gamma is not None, "gamma is None"
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=scheduler_gamma)
else:
    num_training_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, total_iters=1000)

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

def predict_next_solution(model, x, cond):
    y = model.sample(
        torch.randn_like(x[:,1:,:,:]).cuda() * 80.0,
        list(reversed([0.661, 0.9, 5.84, 24.4, 80.0])),
        x,
        z=cond,
    )
    # if args.predict_difference:
    #     y = y * args.difference_weight + x[:,1:,:,:]
    y = y.clamp(-1, 1)
    y = y * 0.5 + 0.5
    y = y * (y_r_max - y_r_min) + y_r_min
    return y

def get_eval_pred(model, model_con, x, cond, strategy, T, times):

    for t in range(T):
        t1 = default_timer()
        # x0 = x * 0.5 + 0.5
        with torch.no_grad():
            im_c = model_con(x, z=cond)
            # im_cr = (im_c - data_min) / (data_max - data_min)
            # im_c = (im_c - 0.5) / 0.5
        im_cc = torch.cat((im_c, x), dim=1)
        im_r = predict_next_solution(model, im_cc, cond)
        im = im_c + im_r
        times.append(default_timer() - t1)
        if t == 0:
            pred = im
        else:
            pred = torch.cat((pred, im), 1)
        if strategy == "markov":
            x = im
            # x = x.clamp(0, 1)
        else:
            x = torch.cat((x[..., 1:, :], im), dim=1)

    return pred

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
total_training_steps = num_training_steps
def output_scale(x, x_out, sigma, sigma_data, sigma_min):
    c_skip = sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)
    c_out = (sigma_data * (sigma - sigma_min)) / (sigma_data**2 + sigma**2) ** 0.5
    c_skip = pad_dims_like(c_skip, x)
    c_out = pad_dims_like(c_out, x)
    return c_skip * x + c_out * x_out

for ep in range(epochs):
    model.train()
    N = math.ceil(math.sqrt((ep * (150 ** 2 - 4) / epochs) + 4) - 1) + 1
    boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).cuda()

    t1 = default_timer()

    train_l2 = loss_ema_l2 = train_sl2 = train_vl2 = 0
    loss_ema = None

    for xx, yy, cond in tqdm(train_loader, disable=not args.verbose):
        loss = 0
        optimizer.zero_grad()
        x = xx.cuda()
        y = yy.cuda()
        cond = cond.cuda()
        # print('x',x.shape)
        # print('x[:, -1:]', x[:, -1:].shape)
        # print('y',y.shape)
        # print('cond',cond.shape)
        # if args.predict_difference:
        #     # Predict difference to next step instead of next step directly.
        #     y = (y - x) / args.difference_weight
            # print('y', y.shape)
        # x0 = (x * 0.5 + 0.5)
        with torch.no_grad():
            im = model_con(x, z=cond)
            # im = (im - 0.5) / 0.5
        im_c = torch.cat((im, x), dim=1)
        y_r = y - im
        y_r = (y_r - y_r_min) / (y_r_max - y_r_min)
        y_r = (y_r - 0.5) / 0.5

        num_timesteps = improved_timesteps_schedule(
            current_training_step,
            total_training_steps,
            initial_timesteps,
            final_timesteps,
        )
        sigmas = karras_schedule(
            num_timesteps, sigma_min, sigma_max, rho, y_r.device
        )
        noise = torch.randn_like(y_r)

        timesteps = lognormal_timestep_distribution(
            y_r.shape[0], sigmas, lognormal_mean, lognormal_std
        )

        current_sigmas = sigmas[timesteps]
        next_sigmas = sigmas[timesteps + 1]

        next_noisy_y = y_r + pad_dims_like(next_sigmas, y_r) * noise
        x_out1 = model(next_noisy_y, next_sigmas, im_c, z=cond)
        next_y = output_scale(next_noisy_y, x_out1, next_sigmas, sigma_data, sigma_min)

        with torch.no_grad():
            current_noisy_y = y_r + pad_dims_like(current_sigmas, y_r) * noise
            x_out2 = model(current_noisy_y, current_sigmas, im_c, z=cond)
            current_y = output_scale(current_noisy_y, x_out2, current_sigmas, sigma_data, sigma_min)


        loss_weights = pad_dims_like(improved_loss_weighting(sigmas)[timesteps], next_y)

        loss = (
                pseudo_huber_loss(next_y, current_y) * loss_weights
        ).mean()

        loss.backward()

        optimizer.step()

        train_l2 += loss.item()
        current_training_step = current_training_step + 1
        update_ema_model_(model, ema_model, ema_decay_rate)

        if not args.step:
            scheduler.step()

    # ema.update()
    if args.step:
        scheduler.step()

    train_times.append(default_timer() - t1)

    # validation
    valid_l2 = valid_vort_l2 = valid_pres_l2 = 0
    valid_loss_by_channel = None
    with torch.no_grad():
        # apply_ema()
        model.eval()
        # model(xx)
        for xx, yy, cond in valid_loader:
            x = xx.cuda()
            y = yy.cuda()
            # x = x * 0.5 + 0.5
            # y = y * 0.5 + 0.5
            cond = cond.cuda()
            # im = model_con(x, z=cond)
            # im_c = torch.cat((im, x), dim=1)

            pred = get_eval_pred(model=model, model_con=model_con, x=x, cond=cond, strategy=args.strategy, T=y.shape[1], times=eval_times)
            valid_l2 += lploss(pred.reshape(len(pred), -1, num_channels), y.reshape(len(y), -1, num_channels)).item()

        # remove_ema()

    t2 = default_timer()
    if args.verbose:
        print(f"Ep: {ep}, time: {t2 - t1}, train: {train_l2 / ntrain}, train_loss_ema: {loss_ema_l2 / ntrain}, valid: {valid_l2 / nvalid}")

    writer.add_scalar("Train/Loss", train_l2 / ntrain, ep)
    writer.add_scalar("Valid/Loss", valid_l2 / nvalid, ep)

    if valid_l2 < best_valid:
        best_epoch = ep
        best_valid = valid_l2
        torch.save(model.state_dict(), path_model)
    if args.early_stopping:
        if ep - best_epoch > args.early_stopping:
            break
stop = default_timer()
train_time = stop - start
train_times = torch.tensor(train_times).mean().item()
num_eval = len(eval_times)
eval_times = torch.tensor(eval_times).mean().item()
model.eval()

# test
model.load_state_dict(torch.load(path_model))
model.eval()
test_l2 = test_vort_l2 = test_pres_l2 = 0
rotations_l2 = 0
reflections_l2 = 0
test_rt_l2 = 0
test_rf_l2 = 0
test_loss_by_channel = None
with torch.no_grad():
    # apply_ema()
    for xx, yy, cond in test_loader:
        x = xx.cuda()
        y = yy.cuda()
        # x = x * 0.5 + 0.5
        # y = y * 0.5 + 0.5
        cond = cond.cuda()
        # im = model_con(x, z=cond)
        # im_c = torch.cat((im, x), dim=1)

        pred = get_eval_pred(model=model, model_con=model_con, x=x, cond=cond, strategy=args.strategy, T=y.shape[1], times=[])
        test_l2 += lploss(pred.reshape(len(pred), -1, num_channels), y.reshape(len(y), -1, num_channels)).item()
    # remove_ema()
    writer.add_scalar("Test/Loss", test_l2 / ntest, best_epoch)

test_time_l2 = test_space_l2 = ntest_super = test_int_space_l2 = test_int_time_l2 = None


print(f"{args.model_type} done training; \nTest: {test_l2 / ntest}, Rotations: {rotations_l2}, Reflections: {reflections_l2}, Super Space Test: {test_space_l2}, Super Time Test: {test_time_l2}")
summary = f"Args: {str(args)}" \
          f"\nParameters: {complex_ct}" \
          f"\nTrain time: {train_time}" \
          f"\nMean epoch time: {train_times}" \
          f"\nMean inference time: {eval_times}" \
          f"\nNum inferences: {num_eval}" \
          f"\nTrain: {train_l2 / ntrain}" \
          f"\nValid: {valid_l2 / nvalid}" \
          f"\nTest: {test_l2 / ntest}" \
          f"\nRotation Test: {test_rt_l2 / ntest}" \
          f"\nReflection Test: {test_rf_l2 / ntest}" \
          f"\nSuper Space Test: {test_space_l2}" \
          f"\nSuper Space Interpolation Test: {test_int_space_l2}" \
          f"\nSuper Time Test: {test_time_l2}" \
          f"\nSuper Time Interpolation Test: {test_int_time_l2}" \
          f"\nBest Valid: {best_valid / nvalid}" \
          f"\nBest epoch: {best_epoch + 1}" \
          f"\nTest Rotation Equivariance loss: {rotations_l2}" \
          f"\nTest Reflection Equivariance loss: {reflections_l2}" \
          f"\nEpochs trained: {ep}"
if swe:
    summary += f"\nVorticity Test: {test_vort_l2 / ntest}" \
               f"\nPressure Test: {test_pres_l2 / ntest}"
txt = "results"
if args.txt_suffix:
    txt += f"_{args.txt_suffix}"
txt += ".txt"

with open(os.path.join(root, txt), 'w') as f:
    f.write(summary)
writer.flush()
writer.close()