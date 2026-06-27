# Dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import xarray as xr

class ERA5ToLDAPSDataset(Dataset):
    def __init__(self, var_dict, topo, target, indices, normalize=True, norm_path="../Data/Dataset", input_vars=None, mask_path=None):
        self.var_dict = var_dict
        self.var_keys = input_vars if input_vars is not None else list(var_dict.keys())
        self.topo = topo  # full high-res topo (H, W)
        self.target = target
        self.indices = indices
        self.normalize = normalize
        self.norm_path = norm_path

        if mask_path is not None:
            self.mask = xr.open_dataset(mask_path)['LAND_surface']  # e.g. LDAPS.landsea_mask.nc
        else:
            self.mask = None

        self.mean = None
        self.std = None
        os.makedirs(norm_path, exist_ok=True)

        if self.normalize:
            mean_path = os.path.join(norm_path, "input_mean.npy")
            std_path = os.path.join(norm_path, "input_std.npy")
            var_path = os.path.join(norm_path, "input_order.txt")

            if os.path.exists(mean_path) and os.path.exists(std_path):
                self.mean = np.load(mean_path)
                self.std = np.load(std_path)
                with open(var_path) as f:
                    assert f.read().splitlines() == self.var_keys, "Variable order mismatch!"
            else:
                self.mean, self.std = self._compute_mean_std()
                np.save(mean_path, self.mean)
                np.save(std_path, self.std)
                with open(var_path, "w") as f:
                    f.write("\n".join(self.var_keys))

        # Target normalization
        self.target_mean_path = os.path.join(norm_path, "target_mean.npy")
        self.target_std_path  = os.path.join(norm_path, "target_std.npy")
        if self.normalize:
            if os.path.exists(self.target_mean_path) and os.path.exists(self.target_std_path):
                self.target_mean = np.load(self.target_mean_path)
                self.target_std  = np.load(self.target_std_path)
            else:
                target_data = self.target.isel(time=indices).values  # (T, H, W)
                if self.mask is not None:
                    land_mask = self.mask.values.astype(bool)  # (H, W)
                    land_mask = land_mask.squeeze(axis=0)
                    valid_target = target_data[:, land_mask]   # → shape: (T, N_land)
                    self.target_mean = valid_target.mean()
                    self.target_std  = valid_target.std()
                else:
                    self.target_mean = target_data.mean()
                    self.target_std  = target_data.std()
                np.save(self.target_mean_path, self.target_mean)
                np.save(self.target_std_path, self.target_std)

    def _compute_mean_std(self):
        means, stds = [], []
        for key in self.var_keys:
            data = self.var_dict[key].values[self.indices]
            means.append(data.mean())
            stds.append(data.std())
        return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # if idx == 0:
            # print("📌 [var_keys & dims check]")
            # for key in self.var_keys:
            #     da = self.var_dict[key]
                #print(f"  - {key}: dims = {da.dims}, shape = {da.shape}")  # check the dimension and name
        i = self.indices[idx]

        x_list = []
        for key in self.var_keys:
            x_i = self.var_dict[key].values[i].astype(np.float32)  # access by position only
            x_list.append(x_i)

        x = np.stack(x_list, axis=0)

        if self.normalize and self.mean is not None and self.std is not None:
            x = (x - self.mean[:, None, None]) / self.std[:, None, None]

        topo_tensor = self.topo.values[np.newaxis, ...].astype(np.float32)  # (1, H_hr, W_hr)
        y = self.target.isel(time=i).values.astype(np.float32)

        if self.normalize and self.target_mean is not None and self.target_std is not None:
            y = (y - self.target_mean) / self.target_std
        
        if self.mask is not None:
            mask_tensor = self.mask.values.astype(np.float32)  # (H_hr, W_hr)
            return torch.tensor(x), torch.tensor(topo_tensor), torch.tensor(y), torch.tensor(mask_tensor)

        return torch.tensor(x), torch.tensor(topo_tensor), torch.tensor(y)
