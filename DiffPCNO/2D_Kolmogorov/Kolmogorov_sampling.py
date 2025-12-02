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

from functools import partial
from utils import pde_data, LpLoss, eq_check_rt, eq_check_rf
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

parser.add_argument("--results_path", type=str, default="/Path_to/results/DiffPCNO/NS/", help="path to store results")
parser.add_argument("--suffix", type=str, default="seed1", help="suffix to add to the results path")
parser.add_argument("--txt_suffix", type=str, default="ns_V1e-3_N10000_T30_DiffPCNO_seed1", help="suffix to add to the results txt")
parser.add_argument("--data_path", type=str, default='/Path_to/Dataset/NS/ns_data4training_zli_samplefreq2e3_dsfreq4.h5', help="path to the data")
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
parser.add_argument("--grid", type=str, default='cartesian', help="[symmetric, cartesian, None]")

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--early_stopping", type=int, default=50, help="stop if validation error does not improve for successive epochs")
parser.add_argument("--batch_size", type=int, default=20)
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

TRAIN_PATH = args.data_path

# FNO data specs
S = Sx = Sy = 64 # spatial res
S_super = 4 * S # super spatial res
T_in = 10 # number of input times
T = args.T
T_super = 4 * T # prediction temporal super res
d = 2 # spatial res
num_channels = 1



# adjust data specs based on model type and data path
threeD = args.model_type in ["FNO3d", "FNO3d_aug",
                             "GCNN3d_p4", "GCNN3d_p4m",
                             "GFNO3d_p4", "GFNO3d_p4m",
                             "radialNO3d_p4", "radialNO3d_p4m",
                             "Unet_Rot_3D"]
extension = TRAIN_PATH.split(".")[-1]
swe = os.path.split(TRAIN_PATH)[-1] == "ShallowWater2D"
rdb = TRAIN_PATH.split(os.path.sep)[-1][:6] == "2D_rdb"
ns_zli = TRAIN_PATH.split("/")[-1][-6:] == "zli.h5"
ns = TRAIN_PATH.split("/")[-1][:2] == "ns"
grid_type = "symmetric"
if args.grid:
    grid_type = args.grid
    assert grid_type in ['symmetric', 'cartesian', 'None']

if rdb:
    assert T == 24, "T should be 24 for rdb"
    T_in = 1
    S = Sx = Sy = 128
    S_super = 128
    T_super = 96
elif swe:
    assert not args.super, "Superresolution not supported for pdearena"
    assert T == 9, "T should be 9 for swe"
    T_in = 2
    Sy, Sx = 96, 192
    num_channels = 2
    grid_type = "cartesian"
elif ns:
    assert T == 20, "T should be 20 for ns"
    T_in = 10
    S = Sx = Sy = 64
    num_channels = 2  # (u, v)
    S_super = 4 * S  # super spatial res
    T_super = 4 * T  # prediction temporal super res
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

root = '/Path_to_pre_trained/Models/2D_Kolmogorov/DiffPCNO'
path_model = os.path.join(root, 'model.pt')
writer = SummaryWriter(root)


################################################################
# Model init
################################################################
if args.model_type in ["Unet"]:
    # model = FNO1d(num_channels=num_channels, hidden_channels=64, param_conditioning=args.param_conditioning, modes=modes, width=width, initial_step=initial_step).cuda()
    model = Unet(n_input_scalar_components=args.n_scalar_components, n_input_vector_components=args.n_vector_components,
         n_output_scalar_components=args.n_scalar_components,
         n_output_vector_components=args.n_vector_components, time_history=args.time_history + args.time_future,
         time_future=args.time_future, hidden_channels=64, activation=args.activation,
         norm=True, param_conditioning=args.param_conditioning).cuda()
else:
    raise NotImplementedError("Model not recognized")

# Define \theta_{-}, which is EMA of the params
ema_model = Unet(n_input_scalar_components=args.n_scalar_components, n_input_vector_components=args.n_vector_components,
         n_output_scalar_components=args.n_scalar_components,
         n_output_vector_components=args.n_vector_components, time_history=args.time_history + args.time_future,
         time_future=args.time_future, hidden_channels=64, activation=args.activation,
         norm=True, param_conditioning=args.param_conditioning).cuda()
ema_model.load_state_dict(model.state_dict())

model_con = PCNO2d(num_channels=num_channels, initial_step=initial_step, modes1=modes, modes2=modes, width=width,
                  grid_type=grid_type).cuda()
path_model_con = 'Path_to/Models/2D_Kolmogorov/PCNO/model.pt'
model_con.load_state_dict(torch.load(path_model_con))
model_con.eval()
################################################################
# load data
################################################################
full_data = None # for superres
ns_full_resolution = 0
ns_zli_read = 0
# print('ns_zli',ns_zli)
# print('ns',ns)
if ns_zli and ns_zli_read:  # incompressible NS data in FNO paper
    print(f'>> Reading in zli full incompressible NS data..')
    assert num_channels == 2, "num channels should be 2 for ns data (two velocity components)"
    assert d == 2, "spatial dim should be 2 for ns data"
    sub = 1
    try:
        with h5py.File(TRAIN_PATH, 'r') as f:
            data = np.expand_dims(np.array(f['u'], dtype=np.float32), axis=-1)
            data = np.concatenate((data, np.expand_dims(np.array(f['v'], dtype=np.float32), axis=-1)), axis=-1)
            # tttt = np.array(f['t'], dtype=np.float32)
        # data = np.transpose(data, (0, 2, 3, 1, 4))
    except:
        data = scipy.io.loadmat(os.path.expandvars(TRAIN_PATH))['velocity'].astype(np.float32)
    # remove the 1st timestep which is a random initialization and is not divergence free
    data = data[..., 1:, :]
    sample_rate = 4
    full_data = data[-ntest:, ..., :(T_in + T) * sample_rate, :]
    # data = data[:, ::sample_rate, ::sample_rate, :30, :]
    # data = data[..., :30, :]

    sampler = torch.nn.AvgPool2d(kernel_size=4)
    data = sampler(torch.tensor(data[..., ::sample_rate, :])[..., :T_in + T, :]
                   .reshape(data.shape[0], S_super, S_super, -1).permute(0, 3, 1, 2)) \
        .permute(0, 2, 3, 1).reshape(data.shape[0], Sx, Sy, -1, num_channels).numpy()

    i_save_downsampled_data = 1
    if i_save_downsampled_data:
        fs = h5py.File('./data/ns_downsampled/ns_data4training_zli_samplefreq2e3_dsfreq%d.h5' % sample_rate, 'w')
        fs.create_dataset('velocity', data=data)
        fs.close()

        fs = h5py.File('./data/ns_downsampled/ns_data4superres_zli_samplefreq2e3.h5', 'w')
        fs.create_dataset('velocity', data=full_data)
        fs.close()
elif ns_full_resolution:  # incompressible NS generated from PDEBench
    print(f'>> Reading in full incompressible NS data..')
    assert num_channels == 2, "num channels should be 2 for ns data (two velocity components)"
    assert d == 2, "spatial dim should be 2 for ns data"
    sub = 1
    try:
        with h5py.File(TRAIN_PATH, 'r') as f:
            data = np.array(f['velocity'], dtype=np.float32)
            # ttt = np.array(f['t'], dtype=np.float32)
        data = np.transpose(data, (0, 2, 3, 1, 4))
    except:
        data = scipy.io.loadmat(os.path.expandvars(TRAIN_PATH))['velocity'].astype(np.float32)
    # remove the 1st timestep which is a random initialization and is not divergence free
    data = data[..., 1:, :]
    sample_rate = 4
    full_data = data[-ntest:, ..., :(T_in + T) * sample_rate, :]
    # data = data[:, ::sample_rate, ::sample_rate, :30, :]
    # data = data[..., :30, :]

    sampler = torch.nn.AvgPool2d(kernel_size=4)
    data = sampler(torch.tensor(data[..., ::sample_rate, :])[..., :T_in + T, :]
                   .reshape(data.shape[0], S_super, S_super, -1).permute(0, 3, 1, 2)) \
        .permute(0, 2, 3, 1).reshape(data.shape[0], Sx, Sy, -1, num_channels).numpy()

    i_save_downsampled_data = 0
    if i_save_downsampled_data:
        fs = h5py.File('./data/ns_downsampled/ns_data4training_zli_dsfreq%d.h5' % sample_rate, 'w')
        fs.create_dataset('velocity', data=data)
        fs.close()

        fs = h5py.File('./data/ns_downsampled/ns_data4superres_zli.h5', 'w')
        fs.create_dataset('velocity', data=full_data)
        fs.close()
elif ns:  # incompressible NS
    print(f'>> Reading in downsampled incompressible NS data..')
    assert num_channels == 2, "num channels should be 2 for ns data (two velocity components)"
    assert d == 2, "spatial dim should be 2 for ns data"
    sub = 1
    train_path_downsampled = './data/ns_data4training_zli_samplefreq2e3_dsfreq4.h5'
    # train_path_downsampled = './data/ns_data4training_zli_samplefreq1e4_dsfreq4.h5'
    # train_path_downsampled = './data/ns_sim_2d-1.h5'

    # train_path_superres = './data/ns_data4superres_zli.h5'
    try:
        with h5py.File(TRAIN_PATH, 'r') as f:
            data = np.array(f['velocity'], dtype=np.float32)
    except:
        data = scipy.io.loadmat(os.path.expandvars(TRAIN_PATH))['velocity'].astype(np.float32)

assert data.shape[-2] >= T + T_in, "not enough time" # ensure there are enough time steps

if not swe:
    data = torch.from_numpy(data)
    # Normalize data to [0, 1] using min-max normalization
    data_min = data.min()
    data_max = data.max()
    data = (data - data_min) / (data_max - data_min)
    # data = (data - 0.5) / 0.5
    print('data_max', data_max)
    print('data_min',data_min)

assert len(data) >= ntrain + nvalid + ntest, f"not enough data; {len(data)}"

train = data[:ntrain]
assert len(train) == ntrain, "not enough training data"

test = data[-ntest:]
test_rt = test.rot90(dims=list(spatial_dims)[:2])
test_rf = test.flip(dims=(spatial_dims[0], ))
assert len(test) == ntest, "not enough test data"

valid = data[-(ntest + nvalid):-ntest]
assert len(valid) == nvalid, "not enough validation data"

# if args.verbose:
print(f"{args.model_type}: Train/valid/test data shape: ")
print(train.shape)
print(valid.shape)
print(test.shape)

assert Sx == train.shape[-3], f"Spatial downsampling should give {Sx} grid points"
assert Sy == train.shape[-4], f"Spatial downsampling should give {Sy} grid points"

train_data = pde_data(train, strategy=args.strategy, T_in=T_in, T_out=T, std=args.noise_std)
ntrain = len(train_data)
valid_data = pde_data(valid, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
nvalid = len(valid_data)
test_data = pde_data(test, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
test_rt_data = pde_data(test_rt, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
test_rf_data = pde_data(test_rf, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
ntest = len(test_data)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
test_rt_loader = torch.utils.data.DataLoader(test_rt_data, batch_size=batch_size, shuffle=False)
test_rf_loader = torch.utils.data.DataLoader(test_rf_data, batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################


y_r_max = float('-inf')
y_r_min = float('inf')
for xx, yy in train_loader:
    x = xx.cuda()
    y = yy.cuda()
    # x0 = x * (data_max - data_min) + data_min
    # y0 = y * (data_max - data_min) + data_min
    x0 = x * (data_max - data_min) + data_min
    with torch.no_grad():
        im = model_con(x0)
        im_r = (im - data_min) / (data_max - data_min)
        # im_r = (im_r - 0.5) / 0.5
        # print('im_r',im_r.shape)
    y_r = y - im_r.squeeze(-2)
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

def predict_next_solution(model, x):
    # m = x[..., :2] + torch.randn_like(x[..., 2:]).cuda() * 80.0
    # print('m',m.shape)
    # print('x',x.shape)
    # x_c = x[..., :2]
    # x_c = (x_c - data_min) / (data_max - data_min)
    # x_c = (x_c - 0.5) / 0.5
    y = model.sample(
        torch.randn_like(x[..., 2:]).cuda() * 80.0,
        list(reversed([0.661, 0.9, 5.84, 24.4, 80.0])),
        x,
    )
    # if args.predict_difference:
    #     y = y * args.difference_weight + x[:,1:,:,:]

    y = y.clamp(-1, 1)
    y = y * 0.5 + 0.5
    y = y * (y_r_max - y_r_min) + y_r_min
    return y

def get_eval_pred_m(model, model_con, x, strategy, T, times, num_samples=10):
    all_preds = []
    all_preds_r = []
    # x0 = x

    for i in range(num_samples):
        pred = None
        pred_r = None
        x0 = x
        for t in range(T):
            t1 = default_timer()
            # x0 = x * 0.5 + 0.5
            with torch.no_grad():
                im_c = model_con(x0)
                im_cr = (im_c - data_min) / (data_max - data_min)
                # im_cr = (im_cr - 0.5) / 0.5
            im_cc = torch.cat((im_c, x0), dim=-1)
            im_r = predict_next_solution(model, im_cc)
            im = im_cr + im_r
            im = im.clamp(0, 1)
            # im = im.unsqueeze(-2)
            # print('im',im.shape)
            times.append(default_timer() - t1)
            if t == 0:
                pred = im
                pred_r = im_r
            else:
                pred = torch.cat((pred, im), -2)
                pred_r = torch.cat((pred_r, im_r), -2)
            if strategy == "markov":
                # x0 = im_c
                x0 = im * (data_max - data_min) + data_min
                # x = x.clamp(0, 1)
            else:
                x0 = torch.cat((x0[..., 1:, :], im), dim=-2)
        all_preds.append(pred.unsqueeze(0))
        all_preds_r.append(pred_r.unsqueeze(0))
    all_preds = torch.cat(all_preds, dim=0)
    all_preds_r = torch.cat(all_preds_r, dim=0)

    mean_pred = all_preds.mean(dim=0)
    # std_pred = all_preds.std(dim=0)
    std_pred_r = all_preds_r.std(dim=0)

    return mean_pred, std_pred_r

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
    print('preds_y', preds_y.shape)

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
def divergence_fft(u, v, domain_size_x=1.0, domain_size_y=1.0):
    B, Sy, Sx = u.shape

    kx = torch.fft.fftfreq(Sx, d=domain_size_x/Sx).to(u.device) * 2 * torch.pi  # shape (Sx,)
    ky = torch.fft.fftfreq(Sy, d=domain_size_y/Sy).to(u.device) * 2 * torch.pi  # shape (Sy,)

    Kx, Ky = torch.meshgrid(kx, ky, indexing='xy')
    Kx = Kx.T
    Ky = Ky.T

    Kx = Kx.unsqueeze(0).expand(B, -1, -1)
    Ky = Ky.unsqueeze(0).expand(B, -1, -1)

    # FFT
    u_ft = torch.fft.fft2(u, dim=(-2,-1))
    v_ft = torch.fft.fft2(v, dim=(-2,-1))

    dudx = torch.fft.ifft2(1j * Kx * u_ft, dim=(-2,-1)).real
    dvdy = torch.fft.ifft2(1j * Ky * v_ft, dim=(-2,-1)).real

    div = dudx + dvdy
    return div


model.load_state_dict(torch.load(path_model))
model.eval()
test_l2 = test_vort_l2 = test_pres_l2 = 0
rotations_l2 = 0
reflections_l2 = 0
test_rt_l2 = 0
test_rf_l2 = 0
test_loss_by_channel = None
total_div0 = momentum_l = 0.0
key = 0
i = 0

with torch.no_grad():
    # apply_ema()
    for xx, yy in test_loader:
        xx = xx.cuda()
        yy = yy.cuda()
        # x = (xx * 0.5 + 0.5)
        x = xx * (data_max - data_min) + data_min
        y = yy * (data_max - data_min) + data_min
        input_data = x
        # x = x * 0.5 + 0.5
        # y = y * 0.5 + 0.5
        # cond = cond.cuda()
        # im = model_con(x, z=cond)
        # im_c = torch.cat((im, x), dim=1)

        pred, pred_std = get_eval_pred_m(model=model, model_con=model_con, x=x, strategy=args.strategy, T=T, times=[], num_samples=50)
        pred = pred * (data_max - data_min) + data_min
        test_l2 += lploss(pred.reshape(len(pred), -1, num_channels), y.reshape(len(y), -1, num_channels)).item()

        print('pred', pred.shape)
        print('pred_std', pred_std.shape)

        x1, x2 = pred[..., 0], pred[..., 1]
        for t in range(pred.shape[3]):
            div0 = divergence_fft(x1[..., t], x2[..., t])
            # print('div0', div0.shape)
            total_div0 += div0.mean()

        # x1, x2 = pred[..., 0], pred[..., 1]
        # div0 = torch.gradient(x1, dim=1)[0] + torch.gradient(x2, dim=2)[0]
        # total_div0 += div0.mean()

        diff = pred.reshape(len(pred), -1, num_channels) - y.reshape(len(y), -1, num_channels)
        l2 = torch.norm(diff, dim=-1)
        momentum_l += l2.mean()
        # Visulazation
        gt_u, gt_v = y[..., 0], y[..., 1]
        pred_u, pred_v = pred[..., 0], pred[..., 1]
        pred_stdu, pred_stdv = pred_std[..., 0], pred_std[..., 1]

        # # gt_hm, gt_um, gt_vm = torch.unsqueeze(gt_h, dim=-1), torch.unsqueeze(gt_u, dim=-1), torch.unsqueeze(gt_v, dim=-1)
        # #U
        # gt_umm = gt_u.permute(0, 3, 1, 2)
        gt_um = torch.rot90(gt_u, k=-1, dims=[1, 2])
        print('gt_um', gt_um.shape)

        # pred_umm = pred_u.permute(0, 3, 1, 2)
        out_um = torch.rot90(pred_u, k=-1, dims=[1, 2])

        # pred_std_umm = pred_stdu.permute(0, 3, 1, 2)
        outm_ustd = torch.rot90(pred_stdu, k=-1, dims=[1, 2])

        print('pred_u', out_um.shape)
        print('outm_ustd', outm_ustd.shape)
        # MOIVE
        movie_dir = '/Path_to/results/DiffPCNO/U/%s/' % (str(i))
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

        #V
        gt_vm = torch.rot90(gt_v, k=-1, dims=[1, 2])
        print('gt_vm', gt_vm.shape)

        # pred_umm = pred_u.permute(0, 3, 1, 2)
        out_vm = torch.rot90(pred_v, k=-1, dims=[1, 2])

        # pred_std_umm = pred_stdu.permute(0, 3, 1, 2)
        outm_vstd = torch.rot90(pred_stdv, k=-1, dims=[1, 2])

        print('pred_v', out_vm.shape)
        print('outm_vstd', outm_vstd.shape)
        movie_dir = '/Path_to/results/DiffPCNO/V/%s/' % (str(i))
        os.makedirs(movie_dir, exist_ok=True)
        movie_name = 'V.gif'
        frame_basename = 'V_frame'
        frame_ext = 'jpg'
        plot_title = ""
        field = 0
        val_cbar_index = -1
        err_cbar_index = -1
        font_size = 12
        remove_frames = True
        generate_movie_2D(key, input_data.cpu(), gt_vm.cpu(), out_vm.cpu(), outm_vstd.cpu(),
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
total_divl0 = total_div0 / (ntest*T)
momentum_l2 = momentum_l / ntest


print(f"{args.model_type} done training; \nTest: {test_l2 / ntest}, mass_loss: {total_divl0}, momuntum_loss: {momentum_l2}, Super Space Test: {test_space_l2}, Super Time Test: {test_time_l2}")

writer.flush()
writer.close()