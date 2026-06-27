import torch
import numpy as np
import xarray as xr
import argparse
import yaml
import os
from models import EDSR, SRCNN
from model_srGAN import GAN_GEN, GAN_DIS, build_condition_for_D
from model_SwinIR_v3 import SwinIR_v3, SwinIR_v3_wo_topo
from Dataloader import get_dataloaders ,update_data_paths_by_resolution
#from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import warnings
import random

warnings.filterwarnings("ignore")

# ---------------------------
# 0. CLI Arguments & Config Load
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="./config.yaml")
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--model', type=str, default=None, help='Override model type in config.yaml')
parser.add_argument('--gpu', type=int, help="override gpu in cofig.yaml")
parser.add_argument('--res', type=float, default=None)
parser.add_argument('--k_fold', type=int, default=0)

args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
if args.res is not None:
    config['res'] = args.res
if args.model is not None:
    config['model'] = args.model.lower()
if args.ckpt is not None:
    config['ckpt'] = args.ckpt
if args.gpu is not None:
    config['gpu'] = args.gpu
if args.k_fold != 0:
    config['k_fold'] = args.k_fold


# ---------------------------
# 1. Device & Random Seed
# ---------------------------
seed = config.get('seed', 42)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

in_channels = len(config['input_vars'])
config['in_channels'] = in_channels


device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
res = float(config['res'])
res_label = f"{res:.1f}".rstrip("0").rstrip(".") + "deg"

output_dir = os.path.join("./outputs", res_label)
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# 2. Load Data
# ---------------------------
#config['num_workers'] = 0
_, _, test_loader, static_dict, full_mask = get_dataloaders(config_dict=config)

def squeeze_static(t: torch.Tensor) -> torch.Tensor:
    # 허용: (H,W), (1,H,W), (B=1,1,H,W)
    if t.dim() == 4 and t.size(0) == 1 and t.size(1) == 1:
        t = t[0, 0]  # -> (H,W)
    if t.dim() == 3 and t.size(0) == 1:
        t = t[0]     # -> (H,W)
    return t

static_dict = {k: squeeze_static(v).to(device) for k, v in static_dict.items()}
full_mask = squeeze_static(full_mask).to(device)  # (H,W), 1/0 mask

STATIC_ORDER = tuple(config.get("static_order", ["elevation", "slope", "mask"])) 
use_lai_lr = bool(config.get("use_lai_lr", True))  # LR lai를 조건/Generator 입력에 포함할지
use_static = bool(config.get("use_static", True))
use_gfs = bool(config.get("use_gfs_input", False))
use_cfs = bool(config.get("use_cfs_input", False))

# ---------------------------
# 3. Load Model
# ---------------------------
model_type = config['model']
upsample_mode=config["upsample_mode"]


res_to_upsteps_map = {0.5: 5, 1.0: 6, 2.0: 7, 3.0: 7}
up_steps = res_to_upsteps_map.get(res, config['upsteps'])


print(f"Model type : {model_type}")
if model_type == "edsr":
    model = EDSR(in_channels=in_channels,
        num_features=config['num_features'],
        num_blocks=config['num_blocks'],
        target_size=(731, 430),
        use_checkpoint=False,    # 메모리 아끼고 싶을 때 True
        shallow_skip=True,       # bilinear 업샘플된 얕은 경로(롱스킵) 더해 안정화
        shallow_from_idx=0   
    )
elif model_type == "srcnn":
    model = SRCNN(in_channels_dyn=in_channels,       # x_dyn의 채널 수 (t2m, u10, v10, sp → 4)
    out_channels=config['out_channels'],          # 출력 채널 수 (기온 1변수면 1)
    target_size=(731, 430),  # 고해상도 타깃 크기 (LDAPS 크기)
    t2m_channel=None,           # x_dyn에서 t2m 채널 index (없으면 None)
    use_coords="auto"        # lat/lon 있으면 사용, 없으면 내부 생성
    )
elif model_type == "srgan":
    model = GAN_GEN(in_channels_dyn=in_channels,
    static_order=STATIC_ORDER,   # ('elevation','slope','mask')
    use_lai_lr=use_lai_lr,      # True/False
    upscale=int(config.get("upscale", 6))
    )
elif model_type == "swinir":
    if use_static == True :
        model =  SwinIR_v3(in_channels_dyn=in_channels, static_channels=5,upsample_mode=upsample_mode, up_steps=up_steps,
                            target_size=(731, 430))
        print("Use static : O")
    else :
        model = SwinIR_v3_wo_topo(in_channels_dyn=in_channels, upsample_mode=upsample_mode, up_steps=up_steps, target_size=(731, 430))

        print("Use static : X")
else:
    raise ValueError("Unsupported model type")

checkpoint = torch.load(f"{config['ckpt']}", map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

# 추론 파일명용 태그 생성
print(f"[INFO] Loaded model={model_type}, ckpt={config['ckpt']}")

model_tag = os.path.splitext(os.path.basename(config['ckpt']))[0]

# ---------------------------
# 4. Inference Loop
# ---------------------------
preds, trues = [], []
test_static_dict = static_dict.copy()

ZERO_OUT_TEST = False
if ZERO_OUT_TEST:
    # 1. 순수 지형 정보 제거
    print("[INFO] zero-out EXP : Topography value input to zero")
    if 'elevation' in test_static_dict:
        test_static_dict['elevation'] = torch.zeros_like(test_static_dict['elevation'])
    if 'slope' in test_static_dict:
        test_static_dict['slope'] = torch.zeros_like(test_static_dict['slope'])

    if 'lai' in test_static_dict:
        test_static_dict['lai'] = torch.zeros_like(test_static_dict['lai'])

        
    print("[INFO] Static variables (elevation, slope, lai) have been zeroed out.")


with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Inference"):
        x = x.to(device)           # (B,C,H,W)
        y = y.to(device).unsqueeze(1) # (B,1,Ht,Wt)
        
        out = model(x, test_static_dict) # (B,1,Ht,Wt)

        # 결과를 CPU로 옮기지 않고 GPU 텐서 그대로 리스트에 저장
        preds.append(out.squeeze(1).detach())      # (B,Ht,Wt) - GPU Tensor
        trues.append(y.squeeze(1).detach())       # (B,Ht,Wt) - GPU Tensor

# ---------------------------
# 5. GPU 메모리 해제 (모델 및 정적 데이터)
# ---------------------------
print("[INFO] Inference finished. Releasing model and static data from GPU...")
del model
del static_dict
#torch.cuda.empty_cache() # 캐시된 GPU 메모리 해제

# full_mask CPU
full_mask_cpu = full_mask.cpu().numpy().astype(bool)

# preds 리스트를 CPU numpy로 변환 + concat
preds_np = [p.detach().cpu().numpy() for p in preds]   # list of (B,H,W)
preds_np = np.concatenate(preds_np, axis=0)            # (T,H,W)

trues_np = [p.detach().cpu().numpy() for p in trues]   # list of (B,H,W)
trues_np = np.concatenate(trues_np, axis=0)  


trues_save = np.where(full_mask_cpu[None, :, :], trues_np, np.nan)
# 바다 = NaN 처리cd 

if use_gfs == True:
    np.save(os.path.join(output_dir, f"pred_gfs_{model_tag}.npy"), preds_save)
    np.save(os.path.join(output_dir, f"true_gfs_{model_tag}.npy"), trues_save)
    print(f"[INFO] Saved preds to {output_dir}/pred_gfs_{model_tag}.npy")
    
elif use_cfs == True:
    np.save(os.path.join(output_dir, f"pred_cfs_{model_tag}.npy"), preds_save)
    
elif ZERO_OUT_TEST == True:
    np.save(os.path.join(output_dir, f"pred_{model_tag}_topo_zero.npy"), preds_save)
else :
    np.save(os.path.join(output_dir, f"pred_{model_tag}.npy"), preds_save)
   # np.save(os.path.join(output_dir, f"true_{model_tag}.npy"), trues_save)

    print(f"[INFO] Saved preds to {output_dir}/pred_{model_tag}.npy")

# ---------------------------
# 6. Metric 계산 (GPU 활용)
# ---------------------------
preds_gpu = torch.cat(preds, dim=0) ;del preds 
trues_gpu = torch.cat(trues, dim=0) ;del trues 


# full_mask (원래 (H,W), 1/0 mask)를 다시 CPU로 가져옴 (저장용)
full_mask_cpu = full_mask.cpu() # 마스크만 다시 로드
# GPU 연산용 마스크 (bool 타입)
mask_gpu = full_mask_cpu.to(device).bool() 


def compute_metrics_gpu(pred_gpu: torch.Tensor, true_gpu: torch.Tensor, mask_gpu: torch.Tensor):
    """
    GPU 텐서를 사용하여 RMSE와 Correlation을 계산합니다.
    pred_gpu/true_gpu: (T,H,W), mask_gpu: (H,W)
    """
    # 마스크 적용 및 1차원 벡터로 변환
    # (T,H,W)에서 (H,W) 마스크를 사용하여 육지 부분만 추출 -> (T * Land_Pixels)
    pv = pred_gpu[:, mask_gpu].flatten()
    tv = true_gpu[:, mask_gpu].flatten()

    if tv.numel() < 2:
        return np.nan, np.nan
    
    # 1. RMSE (Root Mean Square Error)
    rmse = torch.sqrt(torch.mean((pv - tv) ** 2))

    # 2. Pearson Correlation Coefficient (피어슨 상관관계)
    pv_mean = pv.mean()
    tv_mean = tv.mean()

    pv_centered = pv - pv_mean
    tv_centered = tv - tv_mean

    # 분자 (Numerator): 내적 (Inner Product)
    numerator = torch.sum(pv_centered * tv_centered)

    # 분모 (Denominator): 두 벡터의 L2 Norm의 곱
    denom = torch.norm(pv_centered) * torch.norm(tv_centered)

    corr = (numerator / denom) if denom > 0 else torch.tensor(float('nan'), device=device)

    # 결과를 CPU로 이동 후 Python float으로 반환
    return rmse.item(), corr.item()

print("[INFO] Computing metrics on GPU...")
rmse, corr = compute_metrics_gpu(preds_gpu, trues_gpu, mask_gpu)

# ---------------------------
# Save: 바다=NaN로 저장
# ---------------------------
print("[INFO] Preparing to save results...")

# NumPy로 변환하여 저장
#preds_save = np.where(full_mask_cpu.numpy().astype(bool), preds_gpu.cpu().numpy(), np.nan)
#trues_save = np.where(full_mask_cpu.numpy().astype(bool), trues_gpu.cpu().numpy(), np.nan)

# 이전에는 full_mask를 GPU에서 삭제했기 때문에, full_mask_cpu를 사용하여 저장합니다.
#np.save(os.path.join(output_dir, f"pred_{model_tag}.npy"), preds_save)
#np.save(os.path.join(output_dir, f"true_{model_tag}.npy"), trues_save)
print(f"[INFO] Saved npy to {output_dir} (NaN outside land).")

# ---------------------------
# 7. Final Output
# ---------------------------
print(f"✅ RMSE: {rmse:.4f}, Correlation: {corr:.4f}")
