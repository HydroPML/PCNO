import datetime
import os
import random
import numpy as np
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"]="0" # Select the available GPU
from models.FNO import FNO2d
from models.PCNO import PCNO2d

from PIL import Image
import imageio
import matplotlib.pyplot as plt

from utils import pde_data, LpLoss, eq_check_rt, eq_check_rf

import scipy
from timeit import default_timer
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
import h5py
import xarray as xr
from tqdm import tqdm
torch.set_num_threads(1)

def get_eval_pred(model, x, strategy, T, times):

    if strategy == "oneshot":
        pred = model(x)
    else:

        for t in range(T):
            t1 = default_timer()
            im = model(x)
            times.append(default_timer() - t1)
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -2)
            if strategy == "markov":
                x = im
            else:
                x = torch.cat((x[..., 1:, :], im), dim=-2)

    return pred

################################################################
# configs
################################################################

parser = argparse.ArgumentParser()

parser.add_argument("--results_path", type=str, default="/Path_to/results/PCNO/NS/", help="path to store results")
parser.add_argument("--suffix", type=str, default="seed1", help="suffix to add to the results path")
parser.add_argument("--txt_suffix", type=str, default="ns_V1e-3_N10000_T30_PCNO_Test_seed1", help="suffix to add to the results txt")
parser.add_argument("--data_path", type=str, default='/Path_to/Dataset/NS/ns_data4training_zli_samplefreq2e3_dsfreq4.h5', help="path to the data")
parser.add_argument("--super", type=str, default='False', help="enable superres testing")
parser.add_argument("--verbose",type=str, default='True')


parser.add_argument("--T", type=int, default=20, help="number of timesteps to predict")
parser.add_argument("--ntrain", type=int, default=1000, help="training sample size")
parser.add_argument("--nvalid", type=int, default=100, help="valid sample size")
parser.add_argument("--ntest", type=int, default=100, help="test sample size")
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
parser.add_argument("--batch_size", type=int, default=20)
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
                           "UNet2d", "Unet_Rot_M2d", "Unet_Rot_3D"], f"Invalid model type {args.model_type}"
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
                             "PCNO3d"]
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

root = '/Path_to_pre_trained/Models/2D_Kolmogorov/PCNO'
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
else:
    raise NotImplementedError("Model not recognized")


model.load_state_dict(torch.load(path_model))
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
    data_min = data.min()
    data_max = data.max()
    # data = (data - data_min) / (data_max - data_min)
    # data = (data - 0.5) / 0.5

    print('data_max', data.max())
    print('data_min', data.min())

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
# Inferance
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

model.load_state_dict(torch.load(path_model))
model.eval()
test_l2 = test_vort_l2 = test_pres_l2 = 0
rotations_l2 = 0
reflections_l2 = 0
test_rt_l2 = 0
test_rf_l2 = 0
test_loss_by_channel = None
with torch.no_grad():
    for xx, yy in test_loader:
        xx = xx.cuda()
        yy = yy.cuda()
        pred = get_eval_pred(model=model, x=xx, strategy=args.strategy, T=T, times=[]).view(len(xx), Sy, Sx, T, num_channels)
        test_l2 += lploss(pred.reshape(len(pred), -1, num_channels), yy.reshape(len(yy), -1, num_channels)).item()

    writer.add_scalar("Test/Loss", test_l2 / ntest, best_epoch)
    
test_time_l2 = test_space_l2 = ntest_super = test_int_space_l2 = test_int_time_l2 = None
print(f"{args.model_type} done training; \nTest: {test_l2 / ntest}, Rotations: {rotations_l2}, Reflections: {reflections_l2}, Super Space Test: {test_space_l2}, Super Time Test: {test_time_l2}")

writer.flush()
writer.close()