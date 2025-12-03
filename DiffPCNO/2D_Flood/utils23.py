import torch
import os

################################################################
# Dataset class
################################################################
class pde_data(torch.utils.data.Dataset):
    def __init__(self, path_root, T_in, T_out=None, train=True, strategy="markov", rain=False, std=0.0, global_max=26.0):
        self.markov = strategy == "markov"
        self.teacher_forcing = strategy == "teacher_forcing"
        self.one_shot = strategy == "oneshot"
        self.path_root = path_root
        self.p = rain
        # self.data = data[..., :(T_in + T_out)] if self.one_shot else data[..., :(T_in + T_out), :]
        self.data = []
        file_list = [name for name in os.listdir(self.path_root) if os.path.isfile(os.path.join(self.path_root, name))]
        file_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        n = len(file_list)
        print('number', n)
        for name in file_list:
            path_idx = os.path.join(self.path_root, name)
            print('path_idx', path_idx)
            self.data.append(path_idx)
        # max_value = float('-inf')
        # for path in self.data:
        #     tensor_data = torch.load(path)
        #     file_max_value = tensor_data[...,0:1].max().item()
        #     max_value = max(max_value, file_max_value)
        # print('max_value', max_value)
        self.max_value = global_max
        self.nt = T_in + T_out
        self.T_in = T_in
        self.T_out = T_out
        self.num_hist = 1 if self.markov else self.T_in
        self.train = train
        self.noise_std = std

    def log_transform(self, data, eps=1e-2):
        return torch.log(1 + data / eps)

    def __len__(self):
        if self.train:
            if self.markov:
                return len(self.data) * (self.nt - 1)
            if self.teacher_forcing:
                return len(self.data) * (self.nt - self.T_in)
        return len(self.data)

    def __getitem__(self, idx):
        if not self.train or not (self.markov or self.teacher_forcing): # full target: return all future steps
            pde_path = self.data[idx]
            # path_idx = os.path.join(path_root, str(idx) + ".pt")
            pde = torch.load(pde_path)
            pde = pde.permute(1, 2, 0, 3)
            # pde = pde[:-1, :-1, ...]
            # print('pde', pde.shape)
            pde[..., :1] = torch.log1p(pde[..., :1]) / self.max_value
            # pde[..., :1] = (pde[..., :1] - 0.5) / 0.5
            if self.one_shot:
                if self.p:
                    # x = pde[..., :self.T_in, :]
                    x = pde[..., :self.T_in, :]
                    x = x.unsqueeze(-3).repeat([1, 1, self.T_out, 1, 1])
                    y = pde[..., self.T_in:(self.T_in + self.T_out), :]
                else:
                    x = pde[..., :self.T_in, :1]
                    # x = x / self.max_value
                    # x = (x - 0.5) / 0.5
                    # x = h / self.max_value
                    x = x.unsqueeze(-3).repeat([1, 1, self.T_out, 1, 1])
                    y = pde[..., self.T_in:(self.T_in + self.T_out), :1]
            else:
                if self.p:
                    x = pde[..., (self.T_in - self.num_hist):self.T_in, :]
                    y = pde[..., self.T_in:(self.T_in + self.T_out), :]
                else:
                    x = pde[..., (self.T_in - self.num_hist):self.T_in, :1]
                    # x = x / self.max_value
                    # x = (x - 0.5) / 0.5
                    # x = h / self.max_value
                    y = pde[..., self.T_in:(self.T_in + self.T_out), :1]
            return x, y
        pde_idx = idx // (self.nt - self.num_hist) # Markov / teacher forcing: only return one future step
        t_idx = idx % (self.nt - self.num_hist) + self.num_hist
        pde_path = self.data[pde_idx]
        pde = torch.load(pde_path)
        pde = pde.permute(1, 2, 0, 3)
        # pde = pde[:-1, :-1, ...]
        pde[..., :1] = torch.log1p(pde[..., :1]) / self.max_value
        # pde[..., :1] = (pde[..., :1] - 0.5) / 0.5
        # print('pde', pde.shape)
        if self.p:
            x = pde[..., (t_idx - self.num_hist):t_idx, :]
            y = pde[..., t_idx, :1]
        else:
            x = pde[..., (t_idx - self.num_hist):t_idx, :1]
            y = pde[..., t_idx, :1]

        if self.noise_std > 0:
            x += torch.randn(*x.shape, device=x.device) * self.noise_std
        return x, y

################################################################
# Lploss: code from https://github.com/zongyi-li/fourier_neural_operator
################################################################
#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        assert x.shape == y.shape and len(x.shape) == 3, "wrong shape"
        diff_norms = torch.norm(x - y, self.p, 1)
        y_norms = torch.norm(y, self.p, 1)

        if self.reduction:
            loss = (diff_norms/y_norms).mean(-1) # average over channel dimension
            if self.size_average:
                return torch.mean(loss)
            else:
                return torch.sum(loss)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def nse(y_pred, y_true):
    numerator = torch.sum((y_true - y_pred) ** 2)
    denominator = torch.sum((y_true - torch.mean(y_true)) ** 2)
    nse = 1.0 - numerator / denominator
    return nse

def corr(y_pred, y_true):
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    cov = torch.mean((y_true - mean_true) * (y_pred - mean_pred))
    std_true = torch.std(y_true)
    std_pred = torch.std(y_pred)
    cor  = cov / (std_true * std_pred)
    return cor

def critical_success_index(y_pred, y_true, threshold):
    hits = torch.sum(torch.logical_and(y_true > threshold, y_pred > threshold))
    misses = torch.sum(torch.logical_and(y_true > threshold, y_pred <= threshold))
    false_alarms = torch.sum(torch.logical_and(y_true <= threshold, y_pred > threshold))
    csi = hits / (hits + misses + false_alarms + 1e-8)
    return csi

################################################################
# equivariance checks
################################################################
# function for checking equivariance to 90 rotations of a scalar field
def eq_check_rt(model, x, spatial_dims):
    model.eval()
    diffs = []
    with torch.no_grad():
        out = model(x)
        out[out == 0] = float("nan")
        for j in range(len(spatial_dims)):
            for l in range(j + 1, len(spatial_dims)):
                dims = [spatial_dims[j], spatial_dims[l]]
                diffs.append([((out.rot90(k=k, dims=dims) - model(x.rot90(k=k, dims=dims))) / out.rot90(k=k, dims=dims)).abs().nanmean().item() * 100 for k in range(1, 4)])
    return torch.tensor(diffs).mean().item()

# function for checking equivariance to reflections of a scalar field
def eq_check_rf(model, x, spatial_dims):
    model.eval()
    diffs = []
    with torch.no_grad():
        out = model(x)
        out[out == 0] = float("nan")
        for j in spatial_dims:
            diffs.append(((out.flip(dims=(j, )) - model(x.flip(dims=(j, )))) / out.flip(dims=(j, ))).abs().nanmean().item() * 100)
    return torch.tensor(diffs).mean().item()

################################################################
# grids
################################################################
class grid(torch.nn.Module):
    def __init__(self, twoD, grid_type):
        super(grid, self).__init__()
        assert grid_type in ["cartesian", "symmetric", "None"], "Invalid grid type"
        self.symmetric = grid_type == "symmetric"
        self.include_grid = grid_type != "None"
        self.grid_dim = (1 + (not self.symmetric) + (not twoD)) * self.include_grid
        if self.include_grid:
            if twoD:
                self.get_grid = self.twoD_grid
            else:
                self.get_grid = self.threeD_grid
        else:
            self.get_grid = torch.nn.Identity()
    def forward(self, x):
        return self.get_grid(x)

    def twoD_grid(self, x):
        shape = x.shape
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x).reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y).reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        if not self.symmetric:
            grid = torch.cat((gridx, gridy), dim=-1)
        else:
            midx = 0.5
            midy = (size_y - 1) / (2 * (size_x - 1))
            gridx = (gridx - midx) ** 2
            gridy = (gridy - midy) ** 2
            grid = gridx + gridy
        grid = grid.to(x.device)
        return torch.cat((x, grid), dim=-1)

    def threeD_grid(self, x):
        shape = x.shape
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x).reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.linspace(0, 1, size_y).reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.linspace(0, 1, size_z).reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        if not self.symmetric:
            grid = torch.cat((gridx, gridy, gridz), dim=-1)
        else:
            midx = 0.5
            midy = (size_y - 1) / (2 * (size_x - 1))
            gridx = (gridx - midx) ** 2
            gridy = (gridy - midy) ** 2
            grid = torch.cat((gridx + gridy, gridz), dim=-1)
        grid = grid.to(x.device)
        return torch.cat((x, grid), dim=-1)