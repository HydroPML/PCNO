"""
This is a modified version of fourier_2d_time.py from https://github.com/zongyi-li/fourier_neural_operator
"""
import datetime
import os
import random
import numpy as np
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"]="0" # Select the available GPU
from models.PCNO1D import PCNO1d

from PIL import Image
import imageio
import matplotlib.pyplot as plt

from KS.utils import pde_data, LpLoss, eq_check_rt, eq_check_rf # utils is for 1D KSE with fixed viscosity
from loss import CustomMSELoss, PearsonCorrelationScore, ScaledLpLoss


import scipy
import numpy as np
from timeit import default_timer
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
import h5py
import xarray as xr
from tqdm import tqdm
torch.set_num_threads(1)

def get_eval_pred(model, x, z, strategy, T, times):

    for t in range(T):
        t1 = default_timer()
        im = model(x, z=z)
        times.append(default_timer() - t1)
        if t == 0:
            pred = im
        else:
            pred = torch.cat((pred, im), 1)
        if strategy == "markov":
            x = im
        else:
            x = torch.cat((x[..., 1:, :], im), dim=1)

    return pred

################################################################
# configs
################################################################

parser = argparse.ArgumentParser()

parser.add_argument("--results_path", type=str, default="/Path_to/results/KDE/PCNO/", help="path to store results")
parser.add_argument("--suffix", type=str, default="seed1", help="suffix to add to the results path")
parser.add_argument("--txt_suffix", type=str, default="1D_KDE_PCNO_test_seed1", help="suffix to add to the results txt")
parser.add_argument("--super", type=str, default='False', help="enable superres testing")
parser.add_argument("--verbose",type=str, default='True')

parser.add_argument("--T", type=int, default=20, help="number of timesteps to predict")
parser.add_argument("--ntrain", type=int, default=1000, help="training sample size")
parser.add_argument("--nvalid", type=int, default=100, help="valid sample size")
parser.add_argument("--ntest", type=int, default=100, help="test sample size")
parser.add_argument("--nsuper", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--model_type", type=str, default='PCNO1d')
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
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--step", action="store_true", help="use step scheduler")
parser.add_argument("--gamma", type=float, default=None, help="gamma for step scheduler")
parser.add_argument("--step_size", type=int, default=None, help="step size for step scheduler")
parser.add_argument("--lmbda", type=float, default=0.0001, help="weight decay for adam")
parser.add_argument("--strategy", type=str, default="markov", help="markov, recurrent or oneshot")
parser.add_argument("--time_pad", action="store_true", help="pad the time dimension for strategy=oneshot")
parser.add_argument("--noise_std", type=float, default=0.00, help="amount of noise to inject for strategy=markov")
parser.add_argument("--param_conditioning", type=str, default="scalar_3", help="Parameter conditioning")


args = parser.parse_args()

assert args.model_type in ["PCNO1d"], f"Invalid model type {args.model_type}"
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
T_valid = 399
T_test = 399
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
nvalid = args.nvalid # 100
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

root = '/Path_to_pre_trained/Models/1D_KSE/PCNO'
path_model = os.path.join(root, 'model.pt')
writer = SummaryWriter(root)

################################################################
# Model init
################################################################
if args.model_type in ["PCNO1d"]:
    model = PCNO1d(num_channels=num_channels, hidden_channels=64, param_conditioning=args.param_conditioning, modes=modes, width=width, initial_step=initial_step).cuda()

model.load_state_dict(torch.load(path_model))

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

# Path_train = '/Path_to/Dataset/KS/KS_train_fixed_viscosity.h5'
# Path_valid = '/Path_to/Dataset/KS/KS_valid_fixed_viscosity.h5'
# Path_test = '/Path_to/Dataset/KS/KS_test_fixed_viscosity.h5'
Path_train = '/Path_to/Dataset/KS/KS_train_conditional_viscosity.h5'
Path_valid = '/Path_to/Dataset/KS/KS_valid_conditional_viscosity.h5'
Path_test = '/Path_to/Dataset/KS/KS_test_conditional_viscosity.h5'


train = dataset(path=Path_train, mode='train')
test = dataset(path=Path_test, mode='test')
valid = dataset(path=Path_valid, mode='valid')

# if args.verbose:
# print(f"{args.model_type}: Train/valid/test data shape: ")
# print(train.shape)
# print(valid.shape)
# print(test.shape)

train_data = pde_data(train, train=True, strategy=args.strategy, T_in=T_in, T_out=T_train, std=args.noise_std)
ntrain = len(train_data)
valid_data = pde_data(valid, train=False, strategy=args.strategy, T_in=T_in, T_out=T_valid)
nvalid = len(valid_data)
test_data = pde_data(test, train=False, strategy=args.strategy, T_in=T_in, T_out=T_test)
ntest = len(test_data)
# print('ntrain',ntrain)
print('nvalid',nvalid)
print('ntest',ntest)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################
complex_ct = sum(par.numel() * (1 + par.is_complex()) for par in model.parameters())
real_ct = sum(par.numel() for par in model.parameters())
if args.verbose:
    print(f"{args.model_type}; # Params: complex count {complex_ct}, real count: {real_ct}")
writer.add_scalar("Parameters/Complex", complex_ct)
writer.add_scalar("Parameters/Real", real_ct)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.lmbda)
if args.step:
    assert args.step_size is not None, "step_size is None"
    assert scheduler_gamma is not None, "gamma is None"
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=scheduler_gamma)
else:
    num_training_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

lploss = LpLoss(size_average=False)



# test
model.load_state_dict(torch.load(path_model))
model.eval()
corr = PearsonCorrelationScore()
high_corr_thresholds = [0.8, 0.9]
test_l2 = test_vort_l2 = test_pres_l2 = 0
rotations_l2 = 0
reflections_l2 = 0
test_rt_l2 = 0
test_rf_l2 = 0
test_loss_by_channel = None
i = 0
with torch.no_grad():
    for xx, yy, cond in test_loader:
        xx = xx.cuda()
        yy = yy.cuda()
        cond = cond.cuda()
        pred = get_eval_pred(model=model, x=xx, z=cond, strategy=args.strategy, T=yy.shape[1], times=[])
        test_l2 += lploss(pred.reshape(len(pred), -1, num_channels), yy.reshape(len(yy), -1, num_channels)).item()

        y_hm = yy.permute(0, 1, 3, 2)
        pred_hm = pred.permute(0, 1, 3, 2)
        gt_f = y_hm[..., 0]
        pred_f = pred_hm[..., 0]
        if i == 0:
            g_f = gt_f
        else:
            g_f = torch.cat((g_f, gt_f), 0)
        print('g_f', g_f.shape)

        if i == 0:
            p_f = pred_f
        else:
            p_f = torch.cat((p_f, pred_f), 0)
        i = i + 1

    writer.add_scalar("Test/Loss", test_l2 / ntest, best_epoch)

test_time_l2 = test_space_l2 = ntest_super = test_int_space_l2 = test_int_time_l2 = None

corr_timesteps_B = corr(p_f, g_f)
print('corr_timesteps_B', corr_timesteps_B.shape)
corr_timesteps = corr_timesteps_B.mean(0)
# print('corr_timesteps', corr_timesteps)
for threshold in high_corr_thresholds:
    above_thresh = (corr_timesteps > threshold).float()
    high_corr_time = above_thresh.sum().item()
    print(f"High-correlation time (threshold={threshold}):", high_corr_time)



print(f"{args.model_type} done training; \nTest: {test_l2 / ntest}, Rotations: {rotations_l2}, Reflections: {reflections_l2}, Super Space Test: {test_space_l2}, Super Time Test: {test_time_l2}")

writer.flush()
writer.close()