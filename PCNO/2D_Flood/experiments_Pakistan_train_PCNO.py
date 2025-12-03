import datetime
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"]="0" # Select the available GPU
from models.PCNO import PCNO2d

from PIL import Image
import imageio
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils import pde_data, LpLoss, nse, corr, critical_success_index, eq_check_rt, eq_check_rf

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

def get_eval_pred(model, x, y_c, strategy, T, times):
    if strategy == "oneshot":
        pred = model(x)
    else:

        for t in range(T):
            t1 = default_timer()
            im = model(x)
            # im = F.threshold(im, threshold=0, value=0)
            times.append(default_timer() - t1)
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -2)
            if strategy == "markov":
                y_cc = y_c[...,t:(t+1),:]
                x = torch.cat((im, y_cc), dim=-1)
            else:
                x = torch.cat((x[..., 1:, :], im), dim=-2)

    return pred

################################################################
# configs
################################################################

parser = argparse.ArgumentParser()

parser.add_argument("--results_path", type=str, default="/Path_to/PCNO/results/Flood/", help="path to store results")
parser.add_argument("--suffix", type=str, default="seed1", help="suffix to add to the results path")
parser.add_argument("--txt_suffix", type=str, default="Flood_PCNO_Pa_seed1", help="suffix to add to the results txt")
parser.add_argument("--super", type=str, default=False, help="enable superres testing")
parser.add_argument("--verbose",type=str, default=True)
parser.add_argument("--rain",type=str, default=True)

# Flood forecasting with T=24h
parser.add_argument("--T", type=int, default=287, help="number of timesteps to predict")
parser.add_argument("--ntrain", type=int, default=10, help="training sample size")
parser.add_argument("--nvalid", type=int, default=2, help="valid sample size")
parser.add_argument("--ntest", type=int, default=2, help="test sample size")

# Flood forecasting with T=12h
# parser.add_argument("--T", type=int, default=143, help="number of timesteps to predict")
# parser.add_argument("--ntrain", type=int, default=20, help="training sample size")
# parser.add_argument("--nvalid", type=int, default=4, help="valid sample size")

parser.add_argument("--nsuper", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--model_type", type=str, default='PCNO2d')
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--modes", type=int, default=12)
parser.add_argument("--width", type=int, default=20)
parser.add_argument("--Gwidth", type=int, default=10, help="hidden dimension of equivariant layers if model_type=hybrid")
parser.add_argument("--n_equiv", type=int, default=3, help="number of equivariant layers if model_type=hybrid")
parser.add_argument("--reflection", action="store_true", help="symmetry group p4->p4m for data augmentation")
parser.add_argument("--grid", type=str, default='cartesian', help="[symmetric, cartesian, None]")

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--early_stopping", type=int, default=100, help="stop if validation error does not improve for successive epochs")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--step", action="store_true", help="use step scheduler")
parser.add_argument("--gamma", type=float, default=None, help="gamma for step scheduler")
parser.add_argument("--step_size", type=int, default=None, help="step size for step scheduler")
parser.add_argument("--lmbda", type=float, default=0.0001, help="weight decay for adam")
parser.add_argument("--strategy", type=str, default="markov", help="markov, recurrent or oneshot")
parser.add_argument("--time_pad", action="store_true", help="pad the time dimension for strategy=oneshot")
parser.add_argument("--noise_std", type=float, default=0.00, help="amount of noise to inject for strategy=markov")

args = parser.parse_args()

assert args.model_type in ["PCNO2d", "FNO2d",
                           "FNO3d", "FNO3d_aug",
                           "GCNN2d_p4", "GCNN2d_p4m",
                           "GCNN3d_p4", "GCNN3d_p4m",
                           "GFNO2d_p4", "GFNO2d_p4m",
                           "GFNO2d_p4_steer", "GFNO2d_p4m_steer",
                           "GFNO3d_p4", "GFNO3d_p4m",
                           "Ghybrid2d_p4", "Ghybrid2d_p4m",
                           "radialNO2d_p4", "radialNO2d_p4m",
                           "radialNO3d_p4", "radialNO3d_p4m",
                           "Unet_Rot2d", "Unet_Rot_M2d", "Unet_Rot_3D"], f"Invalid model type {args.model_type}"
assert args.strategy in ["teacher_forcing", "markov", "recurrent", "oneshot"], "Invalid training strategy"

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)

data_aug = "aug" in args.model_type

# FNO data specs
Sy = 810
Sx = 441
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

root = args.results_path + f"/{'_'.join(str(datetime.datetime.now()).split())}"
if args.suffix:
    root += "_" + args.suffix
os.makedirs(root)
path_model = os.path.join(root, 'model.pt')
writer = SummaryWriter(root)

################################################################
# Model init
################################################################
if args.model_type in ["PCNO2d"]:
    model = PCNO2d(num_channels=num_channels, initial_step=initial_step, modes1=modes, modes2=modes, width=width,
                  grid_type=grid_type).cuda()
elif args.model_type in ["FNO2d", "FNO2d_aug"]:
    model = FNO2d(num_channels=num_channels, initial_step=initial_step, modes1=modes, modes2=modes, width=width,
                  grid_type=grid_type).cuda()
elif "GCNN2d" in args.model_type:
    reflection = "p4m" in args.model_type
    model = GCNN2d(num_channels=num_channels, initial_step=initial_step, width=width, reflection=reflection).cuda()
elif "GCNN3d" in args.model_type:
    reflection = "p4m" in args.model_type
    model = GCNN3d(num_channels=num_channels, initial_step=initial_step, width=width, reflection=reflection).cuda()
elif "GFNO2d" in args.model_type and "steer" in args.model_type:
    reflection = "p4m" in args.model_type
    model = GFNO2d_steer(num_channels=num_channels, initial_step=initial_step, input_size=S, modes=modes, width=width,
                         reflection=reflection).cuda()
elif "GFNO2d" in args.model_type:
    reflection = "p4m" in args.model_type
    model = GFNO2d(num_channels=num_channels, initial_step=initial_step, modes=modes, width=width,
                   reflection=reflection, grid_type=grid_type).cuda()
elif "GFNO3d" in args.model_type:
    reflection = "p4m" in args.model_type
    model = GFNO3d(num_channels=num_channels, initial_step=initial_step, modes=modes, time_modes=time_modes,
                   width=width, reflection=reflection, grid_type=grid_type, time_pad=args.time_pad).cuda()
elif "Ghybrid2d" in args.model_type:
    reflection = "p4m" in args.model_type
    model = Ghybrid2d(num_channels=num_channels, initial_step=initial_step, modes=modes, Gwidth=args.Gwidth,
                      width=width, reflection=reflection, n_equiv=args.n_equiv).cuda()
elif "radialNO2d" in args.model_type:
    reflection = "p4m" in args.model_type
    model = radialNO2d(num_channels=num_channels, initial_step=initial_step, modes=modes, width=width, reflection=reflection,
                       grid_type=grid_type).cuda()
elif "radialNO3d" in args.model_type:
    reflection = "p4m" in args.model_type
    model = radialNO3d(num_channels=num_channels, initial_step=initial_step, modes=modes, time_modes=time_modes,
                       width=width, reflection=reflection, grid_type=grid_type, time_pad=args.time_pad).cuda()
else:
    raise NotImplementedError("Model not recognized")

# test model on training res and superres data
if args.strategy == "oneshot":
    x_shape = [batch_size, Sy, Sx, T, initial_step, num_channels]
    x_shape_super = [1, S_super, S_super, T_super, initial_step, num_channels]
elif args.strategy == "markov":
    x_shape = [batch_size, Sy, Sx, 1, num_channels]
    x_shape_super = [1, *(S_super, ) * d, 1, num_channels]
else: # strategy == recurrent or teacher_forcing
    x_shape = [batch_size, Sy, Sx, T_in, num_channels]
    x_shape_super = [1, *(S_super, ) * d, T_in, num_channels]

model.train()
x = torch.randn(*x_shape).cuda()
if args.strategy == "recurrent":
    for _ in range(T):
        im = model(x)
        x = torch.cat([x[..., 1:, :], im], dim=-2)
else:
    model(x)
# eq_check_rt(model, x, spatial_dims)
# eq_check_rf(model, x, spatial_dims)
if args.super:
    model.eval()
    with torch.no_grad():
        x = torch.randn(*x_shape_super).cuda()
        model(x)

################################################################
# load data
################################################################
full_data = None # for superres
# The data is saved in the .pt format using torch.save, where the original .tif file is converted and stored as a .pt file.
## Datasets for T=24h
path_train = '/Path_to/Datasets/Pakistan/288/train'
path_val = '/Path_to/Datasets/Pakistan/288/valid'
path_test = '/Path_to/Datasets/Pakistan/288/test'
## Datasets for T=12h
# path_train = '/Path_to/Datasets/Pakistan/144/train'
# path_val = '/Path_to/Datasets/Pakistan/144/valid'
# path_test = '/Path_to/Datasets/Pakistan/144/test'


train_data = pde_data(path_train, strategy=args.strategy, T_in=T_in, T_out=T, rain=args.rain, std=args.noise_std)
ntrain = len(train_data)
valid_data = pde_data(path_val, train=False, strategy=args.strategy, T_in=T_in, T_out=T, rain=args.rain)
nvalid = len(valid_data)
test_data = pde_data(path_test, train=False, strategy=args.strategy, T_in=T_in, T_out=T, rain=args.rain)
# test_rt_data = pde_data(test_rt, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
# test_rf_data = pde_data(test_rf, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
ntest = len(test_data)

print('ntrain', ntrain)
print('nvalid', nvalid)
print('ntest', ntest)

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

best_valid = float("inf")

model.eval()
# if args.verbose:
#     print(f"{args.model_type} pre-train equivariance checks: Rotations - {eq_check_rt(model, x, spatial_dims)}, Reflections - {eq_check_rf(model, x, spatial_dims)}")
start = default_timer()
if args.verbose:
    print("Training...")
step_ct = 0
train_times = []
eval_times = []
for ep in range(epochs):
    model.train()
    t1 = default_timer()

    train_l2 = train_vort_l2 = train_pres_l2 = 0

    for xx, yy in tqdm(train_loader, disable=not args.verbose):
        loss = 0
        xx = xx.cuda()
        yy = yy.cuda()

        if args.strategy == "recurrent":
            for t in range(yy.shape[-2]):
                y = yy[..., t, :]
                # print('xx', xx.shape)
                im = model(xx)
                loss += lploss(im.reshape(len(im), -1, num_channels_y), y.reshape(len(y), -1, num_channels_y))
                xx = torch.cat((xx[..., 1:, :], im), dim=-2)
            loss /= yy.shape[-2]
        else:
            im = model(xx)
            if args.strategy == "oneshot":
                im = im.squeeze(-1)
            # print('im',im.shape)
            # print('yy', yy.shape)
            # im = F.threshold(im, threshold=0, value=0)
            loss = lploss(im.reshape(len(im), -1, num_channels_y), yy.reshape(len(yy), -1, num_channels_y))

        train_l2 += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not args.step:
            scheduler.step()
        writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], step_ct)
        step_ct += 1
    if args.step:
        scheduler.step()

    train_times.append(default_timer() - t1)

    # validation
    valid_l2 = valid_vort_l2 = valid_pres_l2 = 0
    valid_loss_by_channel = None
    with torch.no_grad():
        model.eval()
        # model(xx)
        for xx, yy in valid_loader:
            xx = xx.cuda()
            yy = yy.cuda()
            # print('yy', yy.shape)
            yy_g = yy[..., 0:1]
            yy_c = yy[..., 1:3]
            # print('yy_c', yy_c.shape)
            pred = get_eval_pred(model=model, x=xx, y_c=yy_c, strategy=args.strategy, T=T, times=eval_times).view(len(xx), Sy, Sx, T, num_channels_y)
            valid_l2 += lploss(pred.reshape(len(pred), -1, num_channels_y), yy_g.reshape(len(yy_g), -1, num_channels_y)).item()


    t2 = default_timer()
    if args.verbose:
        print(f"Ep: {ep}, time: {t2 - t1}, train: {train_l2 / ntrain}, valid: {valid_l2 / nvalid}")

    writer.add_scalar("Train/Loss", train_l2 / ntrain, ep)
    writer.add_scalar("Valid/Loss", valid_l2 / nvalid, ep)
    if swe:
        writer.add_scalar("Train Vorticity/Loss", train_vort_l2 / ntrain, ep)
        writer.add_scalar("Train Pressure/Loss", train_pres_l2 / ntrain, ep)
        writer.add_scalar("Valid Vorticity/Loss", valid_vort_l2 / nvalid, ep)
        writer.add_scalar("Valid Pressure/Loss", valid_pres_l2 / nvalid, ep)

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
# if args.verbose:
#     print(f"{args.model_type} post-train equivariance checks: Rotations - {eq_check_rt(model, xx, spatial_dims)}, Reflections - {eq_check_rf(model, xx, spatial_dims)}")

# test
model.load_state_dict(torch.load(path_model))
model.eval()
test_l2 = test_nse = test_corr = test_csi_1 = test_csi_2 = test_csi_3 = test_csi_4 = 0

rotations_l2 = 0
reflections_l2 = 0
test_rt_l2 = 0
test_rf_l2 = 0
test_loss_by_channel = None
with torch.no_grad():
    for xx, yy in test_loader:
        xx = xx.cuda()
        yy = yy.cuda()
        yy_g = yy[..., 0:1]
        yy_c = yy[..., 1:3]
        # print('yy_c', yy_c.shape)
        pred = get_eval_pred(model=model, x=xx, y_c=yy_c, strategy=args.strategy, T=T, times=[]).view(len(xx), Sy, Sx, T, num_channels_y)
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

    writer.add_scalar("Test/Loss", test_l2 / ntest, best_epoch)

test_time_l2 = test_space_l2 = ntest_super = test_int_space_l2 = test_int_time_l2 = None

print(f"{args.model_type} done training for Target domain; \nTest: {test_l2 / ntest}, Test_nse: {test_nse / ntest}, Test_corr: {test_corr / ntest}, Test_csi_1: {test_csi_1 / ntest}, "
      f"Test_csi_2: {test_csi_2 / ntest}, Test_csi_3: {test_csi_3 / ntest}, Test_csi_4: {test_csi_4 / ntest}")
summary = f"Args: {str(args)}" \
          f"\nParameters: {complex_ct}" \
          f"\nTrain time: {train_time}" \
          f"\nMean epoch time: {train_times}" \
          f"\nMean inference time: {eval_times}" \
          f"\nNum inferences: {num_eval}" \
          f"\nTrain: {train_l2 / ntrain}" \
          f"\nValid: {valid_l2 / nvalid}" \
          f"\nTest: {test_l2 / ntest}" \
          f"\nTest_nse: {test_nse/ntest}" \
          f"\nTest_corr: {test_corr/ntest}" \
          f"\nTest_csi_1: {test_csi_1/ntest}" \
          f"\nTest_csi_2: {test_csi_2/ntest}" \
          f"\nTest_csi_3: {test_csi_3/ntest}" \
          f"\nTest_csi_4: {test_csi_4/ntest}" \
          f"\nBest Valid: {best_valid / nvalid}" \
          f"\nBest epoch: {best_epoch + 1}" \
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