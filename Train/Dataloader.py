# dataloader.py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import yaml
import os
import xarray as xr

def update_data_paths_by_resolution(config):
    """
    config['res'], ['input_vars'], ['static_vars'] 기반으로
    경로를 ../Data/Dataset/{res}deg/ 이하로 자동 설정.
    """
    base_dir = config['data_paths']['base_path']
    res_str = f"{config['res']:.1f}".rstrip("0").rstrip(".") + "deg"

    data_processing = config['data_processing_clim']
    updated_paths = {}
    
    use_ldaps_input = bool(config.get('use_ldaps_input', False))
    use_gfs_input = bool(config.get('use_gfs_input', False))
    use_cfs_input = bool(config.get('use_cfs_input', False))

    # dynamic input variables
    for var in config['input_vars']:
        if var == 't2m' and use_ldaps_input:
            updated_paths[var] = os.path.join(base_dir, res_str, f"input__LDAPS_tmpr_data_{data_processing}.pt")
        elif use_gfs_input:
            updated_paths[var] = os.path.join(f"../Data/GFS_data/gfs_input_{var}_data_{data_processing}.pt")
        elif use_cfs_input:
            updated_paths[var] = os.path.join(f"../Data/CFS_data/cfs_input_{var}_data_{data_processing}.pt")
        else:
            updated_paths[var] = os.path.join(base_dir, res_str, f"input_{var}_data_{data_processing}.pt")
            
    # static variables
#   if config.get('use_lai_lr', True):
#       updated_paths['lai'] = os.path.join(base_dir, res_str, config['data_paths']['lai'])  # (1,H_lr,W_lr) or (H_lr,W_lr)

    for var in config['static_vars']:
        if var == "lai" or var == "latlon_LR":
            updated_paths[var] = os.path.join(base_dir, res_str, config['data_paths'][var])
        else :
            updated_paths[var] = os.path.join(base_dir, config['data_paths'][var])

    updated_paths['target'] = os.path.join(base_dir, f"target_tmpr_data_{data_processing}.pt")
    return updated_paths

class DownscalingDataset(Dataset):
    def __init__(self, var_dict, target, indices, input_vars):
        self.var_dict = var_dict  # dynamic variables
        self.target = target      # LDAPS-resolution target
        self.indices = indices
        self.input_vars = input_vars

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        dyn_inputs = [self.var_dict[var][t] for var in self.input_vars]  # (H_d, W_d)
        x_dyn = torch.stack(dyn_inputs, dim=0)  # (C_dyn, H_d, W_d)
        y = self.target[t]  # (H_s, W_s)
        return x_dyn.float(), y.float()

# def load_mask(mask_path):
#     ds = xr.open_dataset(mask_path)
#     mask = ds["mask"].squeeze().values  # (H_s, W_s)
#     return torch.tensor(mask).float()


def get_time_indices_by_period(length, start, freq="D",k_fold=0):
    if not (0 <= k_fold <= 5):
        raise ValueError("k_fold must be in [0, 5].")

    # 타임스탬프/연도 벡터화
    t = pd.date_range(start=start, periods=length, freq=str(freq))
    y = t.year.values  # np.ndarray

    # k별 (val_years, test_years)
    if   k_fold == 0: vy = (2021, 2022); ty = (2023, 2024)
    elif k_fold == 1: vy = (2019, 2020); ty = (2021, 2022)
    elif k_fold == 2: vy = (2017, 2018); ty = (2019, 2020)
    elif k_fold == 3: vy = (2015, 2016); ty = (2017, 2018)
    elif k_fold == 4: vy = (2013, 2014); ty = (2015, 2016)
    else:             vy = (2023, 2024); ty = (2013, 2014)  # k=5

    val_mask  = (y >= vy[0]) & (y <= vy[1])
    test_mask = (y >= ty[0]) & (y <= ty[1])
    train_mask = ~(val_mask | test_mask)

    val_idx   = np.where(val_mask)[0].tolist()
    test_idx  = np.where(test_mask)[0].tolist()
    train_idx = np.where(train_mask)[0].tolist()

    # 방어 코드
    # if not train_idx or not val_idx or not test_idx:
    #     raise RuntimeError("Empty split detected. Check time range / start / freq / data length.")
    if not test_idx: 
        print(f"⚠️ [Warning] Empty Test Split for K-Fold {k_fold}. Check data range.")
        if k_fold == 5 and start.startswith("2015"):
             raise RuntimeError("❌ K-Fold 5 targets 2013-2014. GFS data (starts 2015) cannot cover this period.")

    print("[DEBUG] Dataset - K-fold :", k_fold)
    tr_y = f"{t[train_idx[0]].year}~{t[train_idx[-1]].year}" if train_idx else "None"
    val_y = f"{t[val_idx[0]].year}~{t[val_idx[-1]].year}" if val_idx else "None"
    te_y = f"{t[test_idx[0]].year}~{t[test_idx[-1]].year}" if test_idx else "None"
    
    print(f"[DEBUG] Dataset - Train: {tr_y} (#:{len(train_idx)}), Val: {val_y} (#:{len(val_idx)}), Test: {te_y} (#:{len(test_idx)})")
    return train_idx, val_idx, test_idx

def get_dataloaders(config_dict=None, config_path=None):
    if config_dict is not None:
        config = config_dict
    elif config_path is not None:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("Either config_dict or config_path must be provided.")

    input_vars = config["input_vars"]
    static_vars = config["static_vars"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    k_fold = config["k_fold"]

    # resolution에 따라 경로 업데이트
    data_paths = update_data_paths_by_resolution(config)
    print("🔍 [DEBUG] Training Data paths:")
    for var, path in data_paths.items():
        print(f"  - {var}: {path}")

    # variable load
    var_dict = {var: torch.load(data_paths[var],  weights_only=True) for var in input_vars}
    topo_dict = {var: torch.load(data_paths[var],  weights_only=True) for var in static_vars}
    target = torch.load(data_paths['target'],  weights_only=True)
    mask = torch.load(data_paths['mask'],  weights_only=True)
    use_gfs_input = bool(config["use_gfs_input"])
    
    start_date_for_index = config.get("time", {}).get("start_date", "2013-01-01") 
    freq = config.get("time", {}).get("frequency", "D")

    if use_gfs_input:
        gfs_offset = 744  
      
        if target.shape[0] > gfs_offset:
            target_orig_len = target.shape[0]
            target = target[gfs_offset:]
            print(f" [GFS Inference Mode]  ✂️ Sliced Target: {target_orig_len} -> {target.shape[0]} (Removed first {gfs_offset} days)")
        else:
            raise ValueError("❌ Target data is too short for GFS alignment!")

        min_len = min([v.shape[0] for v in var_dict.values()] + [target.shape[0]])
        
        target = target[:min_len]
        for v in var_dict:
            var_dict[v] = var_dict[v][:min_len]

        start_date_for_index = "2015-01-15"
        print(f"✅ Data Aligned. New Start: 2015-01-15 (Total: {min_len})")

        # 3. [중요] 인덱스 생성용 시작 날짜를 2015-01-15로 변경!
        
    print(f"📅 Generating indices from start date: {start_date_for_index}")
    train_idx, val_idx, test_idx = get_time_indices_by_period(target.shape[0], start=start_date_for_index, freq=freq, k_fold=k_fold)
    
    # 4. Dataset 구성
    train_set = DownscalingDataset(var_dict, target, train_idx, input_vars)
    val_set = DownscalingDataset(var_dict, target, val_idx, input_vars)
    test_set = DownscalingDataset(var_dict, target, test_idx, input_vars)

    # 5. DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=False)

    return train_loader, val_loader, test_loader, topo_dict, mask
