import torch
import numpy as np
import yaml
import argparse
from tqdm import tqdm
import os

from Dataloader import get_dataloaders
from model_SwinIR_v3 import SwinIR_v3


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./config.yaml")
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--res", type=float, default=None)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

if args.res is not None:
    config["res"] = float(args.res)
if args.gpu is not None:
    config["gpu"] = args.gpu

res = float(config['res'])
res_label = f"{res:.1f}".rstrip("0").rstrip(".") + "deg"

output_dir = os.path.join("./outputs", res_label)
os.makedirs(output_dir, exist_ok=True)
device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------
# 1) static_dict / full_mask만 dataloader에서 가져오기
# ----------------------------------------------------
_, _, _, static_dict, full_mask = get_dataloaders(config_dict=config)

for k in static_dict:
    static_dict[k] = static_dict[k].float().to(device)
full_mask = full_mask.float().to(device)

res_to_upsteps_map = {0.5: 5, 1.0: 6, 2.0: 7, 3.0: 7}
up_steps = res_to_upsteps_map.get(res, config['upsteps'])

# SwinIR_v3 model (static_channels=6)
model = SwinIR_v3(
    in_channels_dyn=len(config["input_vars"]),
    static_channels=5,
    upsample_mode=config["upsample_mode"],
    up_steps=up_steps,
    target_size=(731,430)
).to(device)

ckpt = torch.load(args.ckpt, map_location=device)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()

# ----------------------------------------------------
# 2) Paleo dyn(LR) 입력 로드
# ---------------------------------------------------- 

period = "paleo"   # "paleo" or "future"
Clims = "pi"
print("[INFO] Model period :", period)
print("[INFO] Climatology :", Clims )

if period == "paleo":
    paleo_dir = config["data_paths"]["paleo_path"]
elif period == "future":
    paleo_dir = config["data_paths"]["future_path"]

if Clims == "clim":
    p_T = torch.load(f"{paleo_dir}/input_TREFHT_data_residual_f_clim.pt")
else:
    p_T = torch.load(f"{paleo_dir}/input_TREFHT_data_residual_f_pi.pt")
p_U = torch.load(f"{paleo_dir}/input_U_data_residual_f.pt")
p_V = torch.load(f"{paleo_dir}/input_V_data_residual_f.pt")
p_P = torch.load(f"{paleo_dir}/input_PS_data_residual_f.pt")

paleo_dyn = torch.stack([p_T, p_U, p_V, p_P], dim=1)  # (T,4,6,4)
paleo_dyn = paleo_dyn.float()

# ----------------------------------------------------
# 3) static_dict 정렬
# ----------------------------------------------------
x_static = {
    "elevation": static_dict["elevation"],
    "slope":     static_dict["slope"],
    "mask":      static_dict["mask"],
    "latlon_LR": static_dict["latlon_LR"],
    "latlon_HR": static_dict["latlon_HR"],
    "lai":       static_dict["lai"],
}


# ----------------------------------------------------
# 4) Paleo inference
# ----------------------------------------------------
preds = []

with torch.no_grad():
    for t in tqdm(range(paleo_dyn.shape[0]), desc="[Paleo Inference]"):
        x_dyn = paleo_dyn[t:t+1].to(device)    # (1,4,6,4)
        out = model(x_dyn, x_static)           # (1,1,731,430)
        preds.append(out.squeeze().cpu().numpy())

preds = np.stack(preds, axis=0)  # (T,731,430)
#np.save("paleo_preds.npy", preds)
np.save(os.path.join(output_dir, f"{period}_preds_{Clims}_f.npy"), preds)
print(f"Saved {period}_preds_{Clims}_f.npy")