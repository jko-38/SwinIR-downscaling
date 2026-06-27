from math import e
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch import nn, optim
from Dataloader import get_dataloaders
from model_SwinIR_v3 import SwinIR_v3, SwinIR_v3_wo_topo
from loss import ssim_loss, grad_loss
import yaml
import argparse
import numpy as np
import os
from tqdm import tqdm
import wandb
from torchvision.utils import save_image
import random
from torchinfo  import summary

# ---------------------------
# 0. CLI & Config
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config.yaml')
parser.add_argument('--res', type=float, default=None, help="Input resolution (e.g., 0.25, 0.5, 1, 2, 3)")
parser.add_argument('--gpu', type=int, default=None,help="override gpu in cofig.yaml")
parser.add_argument('--kfold', type=int, default=None,help="override gpu in cofig.yaml")
parser.add_argument('--model', type=str, default='swinir',help="override gpu in cofig.yaml")
parser.add_argument('--version', type=str, default=None,help="override gpu in cofig.yaml")

args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
if args.res is not None:
    config['res'] = args.res
if args.gpu is not None:
    config['gpu'] = args.gpu
if args.version is not None:
    config['version'] = args.version
if args.kfold is not None:
    config['k_fold'] = args.kfold
if args.model != "swinir":
    config['model'] = args.model

in_channels = len(config["input_vars"])
res = float(config['res'])
res_label = f"{res:.1f}".rstrip("0").rstrip(".") + "deg"

res_to_upsteps_map = {0.5: 5, 1.0: 6, 2.0: 7, 3.0: 7}
up_steps = res_to_upsteps_map.get(res, config['upsteps'])

# Seed 고정
seed = config['seed']
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

epochs = config['epochs']
loss_name = config.get("criterion", "mae")
k_fold=config["k_fold"]
data_m=config['data_processing_clim']
upsample_mode=config["upsample_mode"]
version = config['version']
num_warmup_epochs = config.get('warmup_epochs', 10)
schedule = config.get('scheduler', 'ReduceLROnPlateau')

config["model"]= args.model; name_m = config['model']
if name_m == "swinir":
    model_na = 'SwinIR'
elif name_m == "swinir_v3":
    model_na = 'SwinIR_v3'
else:
    raise ValueError(f"Unknown model: {name_m}")

use_static = bool(config.get("use_static", False))
use_ldaps = bool(config.get("use_ldaps_input", False))
tag_parts = [name_m, str(k_fold), res_label, data_m, version]
varname = "850hPa" if "u850" in config["input_vars"] else "10m"

if any(v in config["input_vars"] for v in ["t850","t925"]):
    tag_parts.append("Tplev")
if use_ldaps:
    tag_parts.append("ldaps")
if not use_static:
    tag_parts.append("topoX")
tag = "_".join(tag_parts)

# ---------------------------
# 1. wandb Init
# ---------------------------
wandb.init(project=config.get("wandb_project", "default_project"),name=tag, config=config,settings=wandb.Settings(init_timeout=500,reinit=True))

# ---------------------------
# 2. Dataloader
# ---------------------------
device = torch.device(f"cuda:{config.get('gpu')}" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, _, static_dict, mask = get_dataloaders(config_dict=config)
static = {k: v.to(device) for k, v in static_dict.items()}
mask = mask.to(device)

# ---------------------------
# 3. Model & Loss
# ---------------------------

if name_m == "swinir":
    if use_static == True:
        model = SwinIR_v3(in_channels_dyn=in_channels, static_channels=5, upsample_mode=upsample_mode, up_steps=up_steps, target_size=(731, 430)).to(device)
    elif use_static == False:
        model_na = "swinir_topoX"
        model = SwinIR_v3_wo_topo(in_channels_dyn=in_channels, upsample_mode=upsample_mode, up_steps=up_steps, target_size=(731, 430)).to(device)
else:
    raise ValueError("Use --model swinir (v3만 지원)")
del static_dict

optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'],betas=(0.9, 0.999))

if schedule == "CosineAnnealingLR":
    warmup_scheduler = LinearLR(optimizer, start_factor=0.001, total_iters=num_warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=(epochs - num_warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[num_warmup_epochs])
else:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=config.get('patience', 5))

crit_name     = config['criterion']              # e.g., 'mae', 'mae+grad+ssim', 'mse+ssim' ...
ssim_lambda   = float(config.get("ssim_lambda", 0.1))
grad_lambda   = float(config.get("grad_lambda", 0.2))
eps           = 1e-8

print(f"[DEBUG] Use Topography data: {use_static}")
print(f"[DEBUG] Model architecture : {model_na}")
print(f"[DEBUG] Upsampling mode : {upsample_mode}")
print("[DEBUG] resolution : ", config.get("res"))
print("[DEBUG] loss function : ", crit_name)


# --- base criterion: mask 구하기위해 sum으로 계산 ---
if crit_name.startswith('mae'):
    base_crit = nn.L1Loss(reduction='sum')
    base_name = "MAE"
    wandb_base_key = "MAE_loss"
elif crit_name.startswith('mse'):
    base_crit = nn.MSELoss(reduction='sum')
    base_name = "MSE"
    wandb_base_key = "MSE_loss"
else:
    # 커스텀 기준이면 sum 반환한다고 가정
    base_crit = nn.L1Loss(reduction='sum')
    base_name = "BASE"
    wandb_base_key = "BASE_loss"

# ---------------------------
# 4. Train Loop
# ---------------------------
ckpt_dir = os.path.join("./checkpoints", f"{res_label}")
os.makedirs(ckpt_dir, exist_ok=True)
train_losses, val_losses = [], []
print("[DEBUG] static inputs :", static.keys())

# ---- 속도/빈도 토글 (원하면 조절) ----
METRIC_EVERY_EPOCHS     = config["METRIC_EVERY_EPOCHS"]    # 몇 에폭마다 샘플 메트릭 계산할지 (RMSE/PCorr/SSIM)
METRIC_VAL_MAX_BATCHES  = config["METRIC_VAL_MAX_BATCHES"]    # 검증에서 메트릭 계산할 최대 배치 수
VIS_EVERY_EPOCHS        = config["VIS_EVERY_EPOCHS"]    # 몇 에폭마다 시각화 저장할지 (0/None이면 비활성)
VIS_VAL_MAX_SAMPLES     = config["VIS_VAL_MAX_SAMPLES"]    # 저장할 샘플 개수(검증에서 앞 샘플)


def prep_mask_for_batch(m, x):
    """mask를 배치 크기에 맞게 broadcast하고 float(0/1) 보장"""
    if m.dim() == 2:   m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3: m = m.unsqueeze(0)
    if m.shape[0] == 1 and x.shape[0] > 1:
        m = m.expand(x.shape[0], -1, -1, -1)
    return m.float()

def pattern_corr_batch(pred, y, m, eps=1e-8):
    """
    배치 단위 spatial pattern correlation.
    각 샘플의 corr을 유효 픽셀수로 가중 평균해 (num, den) 반환.
    """
    B = pred.size(0)
    a = (pred * m).view(B, -1)
    b = (y    * m).view(B, -1)
    w = m.view(B, -1)
    w_sum = w.sum(dim=1) + eps

    a_mean = (a.sum(dim=1, keepdim=True) / w_sum.unsqueeze(1))
    b_mean = (b.sum(dim=1, keepdim=True) / w_sum.unsqueeze(1))

    mask_bin = (w > 0).float()
    a0 = (a - a_mean) * mask_bin
    b0 = (b - b_mean) * mask_bin

    num = (a0 * b0).sum(dim=1)
    den = torch.sqrt((a0*a0).sum(dim=1) * (b0*b0).sum(dim=1) + eps)
    r = (num / den).clamp(-1.0, 1.0)  # (B,)
    return (r * w_sum).sum().item(), w_sum.sum().item()

def _to_vis_img(t):
    """(1,H,W) -> 0~1 min-max(퍼센타일) 정규화"""
    t = t.clone()
    mn = torch.quantile(t, 0.01)
    mx = torch.quantile(t, 0.99)
    t = torch.clamp((t - mn) / (mx - mn + 1e-8), 0, 1)
    t = torch.flip(t, dims=[-2])
    return t

def save_visuals(pred, y, m, out_dir, prefix, max_n=4, use_wandb=True, draw_gt=False):
    """
    pred,y,m: (B,1,H,W). 마스크 밖은 0 처리해서 저장.
    """
    os.makedirs(out_dir, exist_ok=True)
    B = min(pred.size(0), max_n)
    for i in range(B):
        p = (pred[i:i+1] * m[i:i+1])
        g = (y[i:i+1]    * m[i:i+1])
        e = torch.abs(p - g)

        pv = _to_vis_img(p[0])
        gv = _to_vis_img(g[0])
        ev = _to_vis_img(e[0])

    # wandb 로깅할 데이터 초기화
        wandb_log_data = {}
        if use_wandb:
            wandb_log_data[f"{prefix}/pred_{i}"] = wandb.Image(pv.cpu())
            wandb_log_data[f"{prefix}/err_{i}"] = wandb.Image(ev.cpu())

        # 💡 2. draw_gt가 True일 때만 GT 관련 작업 수행
        if draw_gt:
            gv = _to_vis_img(g[0]) 
            save_image(gv, os.path.join(out_dir, f"{prefix}_gt_{i}.png")) 
            if use_wandb:
                wandb_log_data[f"{prefix}/gt_{i}"] = wandb.Image(gv.cpu()) # GT wandb 로깅

        # 3. 수집된 데이터 wandb에 한 번에 로깅
        if use_wandb:
            wandb.log(wandb_log_data, step=epoch+1)

try:
    shape_lr = config['input_shape'][res]
    H_lr, W_lr = shape_lr[0], shape_lr[1]  # 👈 두 인자(H, W)를 가져옴
    print(f"[DEBUG] Input shape: {res}deg - Shape ({H_lr}, {W_lr})")
except KeyError:
    raise KeyError(
        f"'{res}'에 해당하는 'input_shape'가 config.yaml에 없습니다."
        f" (키가 문자열이어야 합니다. 예: '0.5', '1')"
    )


best_val = float("inf")
dummy_input = torch.randn(1, in_channels, H_lr, W_lr).to(device)  # low-res input 크기에 맞게
print("[DEBUG] Model Summary ")
summary(model,
        input_data=(dummy_input, static),
        device=device)

global_step = 0
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    total_num_loss = total_num_base = total_num_grad = total_num_ssim = 0.0
    total_den = 0.0

    for x, y in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training", leave=False):
        x = x.to(device); y = y.to(device).unsqueeze(1)
        m = prep_mask_for_batch(mask, x)

        # ✅ AMP (H100 권장: BF16)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda',  dtype=torch.bfloat16): #, enabled=False):
            pred = model(x, static)

        pred_f = pred.float()
        y_f    = y.float()
        m_f    = m.float()
        
        valid = m_f.sum() + eps
        base_sum = base_crit(pred_f * m_f, y_f * m_f)  # sum
        base = base_sum / valid

        loss = base
        grad_term = None
        ssim_term = None

        if "grad" in crit_name:
            grad_term = grad_loss(pred_f, y_f, m_f, p=1)  # mask-aware
            loss = loss + grad_lambda * grad_term

        if "ssim" in crit_name:
            ssim_term = (1.0 - ssim_loss(pred_f * m_f, y_f * m_f))
            loss = loss + ssim_lambda * ssim_term

        #optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1

        w = valid.item()
        total_num_loss     += loss.item() * w
        total_num_base     += base.item() * w
        if grad_term is not None: total_num_grad     += grad_term.item() * w
        if ssim_term is not None: total_num_ssim += ssim_term.item() * w
        total_den          += w
        
    train_loss     = total_num_loss / max(total_den, 1.0)
    avg_base       = total_num_base / max(total_den, 1.0)  # = MAE/MSE(마스크 평균)
    avg_grad       = (total_num_grad / max(total_den, 1.0)) if "grad" in crit_name else 0.0
    avg_ssim_term  = (total_num_ssim / max(total_den, 1.0)) if "ssim" in crit_name else 0.0
    train_losses.append(train_loss)
    
    
    # =============== VALIDATION ===============
    # =============== VALIDATION ===============
    model.eval()
    val_num_loss =  val_num_base = val_num_grad = val_num_ssimterm = val_den = 0.0
    draw_gt = (epoch == 0)

    # 메트릭(속도절약): 간헐 + 일부 배치만
    do_metrics = ((epoch + 1) % METRIC_EVERY_EPOCHS == 0)
    metric_batches_done = 0
    val_mse_sum = 0.0                      # -> RMSE
    val_ssim_metric_sum = 0.0              # -> SSIM(↑)
    val_pcorr_num = 0.0; val_pcorr_den = 0.0
    
    do_vis = (VIS_EVERY_EPOCHS and (epoch + 1) % VIS_EVERY_EPOCHS == 0)
    vis_done = False
    
    with torch.no_grad():
        for bidx, (x, y) in enumerate(tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validation")):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            m = prep_mask_for_batch(mask, x)

            pred = model(x, static)
            valid = m.sum() + eps
            w = valid.item()

            # --- base val loss ---
            
            base_sum = base_crit(pred * m, y * m)  # sum
            base = base_sum / valid
            loss_val = base

            if "grad" in crit_name:
                grad_term = grad_loss(pred, y, m, p=1)
                loss_val = loss_val + grad_lambda * grad_term
                val_num_grad += grad_term.item() * w
            ssim_metric = None
            if "ssim" in crit_name:
                ssim_metric = ssim_loss(pred * m, y * m)
                ssim_term = (1.0 - ssim_metric)
                loss_val = loss_val + ssim_lambda * ssim_term
                val_num_ssimterm += ssim_term.item() * w

            val_num_loss += loss_val.item() * w
            val_num_base += base.item() * w
            val_den      += w

            if do_metrics and metric_batches_done < METRIC_VAL_MAX_BATCHES:
                diff = (pred - y) * m
                val_mse_sum += (diff * diff).sum().item()

                if ssim_metric is None:
                    ssim_metric = ssim_loss(pred * m, y * m)
                val_ssim_metric_sum += ssim_metric.item() * w

                num, den = pattern_corr_batch(pred, y, m, eps=eps)
                val_pcorr_num += num; val_pcorr_den += den

                metric_batches_done += 1

            # --- 시각화 ---
            if do_vis and (not vis_done):
                vis_dir = os.path.join(ckpt_dir, "visuals", f"ep{epoch+1}")
                save_visuals(pred.detach(), y.detach(), m, vis_dir, prefix="val",
                            max_n=VIS_VAL_MAX_SAMPLES, use_wandb=True, draw_gt=draw_gt)
                vis_done = True

    val_loss   = val_num_loss / max(val_den, 1.0)
    val_base   = val_num_base / max(val_den, 1.0)
    val_grad   = (val_num_grad / max(val_den, 1.0)) if "grad" in crit_name else 0.0
    val_ssim_t = (val_num_ssimterm / max(val_den, 1.0)) if "ssim" in crit_name else 0.0
    val_losses.append(val_loss)
    
    
    if schedule == "ReduceLROnPlateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()

    # --- 메트릭 집계 ---
    if do_metrics and val_pcorr_den > 0:
        val_rmse = (val_mse_sum / val_pcorr_den) ** 0.5
        val_ssim =  val_ssim_metric_sum / val_pcorr_den
        val_pcorr = val_pcorr_num / val_pcorr_den
    else:
        val_rmse = None; val_ssim = None; val_pcorr = None

    # ---- PRINT ----
    line = (f"[Epoch {epoch+1}] " f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} " f"| {base_name}(Train/Val): {avg_base:.5f}/{val_base:.5f} ")
    if "grad" in crit_name: line += f"| Grad(Train/Val): {avg_grad:.4f}/{val_grad:.4f} "
    if "ssim" in crit_name: line += f"| (1-SSIM)(Train/Val): {avg_ssim_term:.4f}/{val_ssim_t:.4f} "
    if val_rmse is not None: line += f"| RMSE(Val*): {val_rmse:.4f} "
    if val_pcorr is not None: line += f"| PCorr(Val*): {val_pcorr:.3f} "
    if val_ssim is not None: line += f"| SSIM(Val*): {val_ssim:.5f} "
    line += f"| LR: {optimizer.param_groups[0]['lr']:.6f}"
    print(line, flush=True)

    # ---- WANDB ----
    log_payload = {
        "train_loss": train_loss, "val_loss": val_loss,
        wandb_base_key: avg_base, f"Val_{wandb_base_key}": val_base,
        "LR": optimizer.param_groups[0]['lr'], }
    if "grad" in crit_name:
        log_payload.update({"GRAD_loss": avg_grad, "Val_GRAD_loss": val_grad})
    if "ssim" in crit_name:
        log_payload.update({"SSIM_loss_term": avg_ssim_term, "Val_SSIM_loss_term": val_ssim_t})
    if val_rmse is not None:
        log_payload.update({"val_rmse_sampled": val_rmse})
    if val_pcorr is not None:
        log_payload.update({"val_pcorr_sampled": val_pcorr})
    if val_ssim is not None:
        log_payload.update({"val_ssim_sampled": val_ssim})
    wandb.log(log_payload,step=epoch+1)

    # ---- CHECKPOINT ----
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{tag}_latest.pt"))
    if val_loss <= (min(val_losses) if len(val_losses) > 0 else float("inf")):
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{tag}_best.pt"))
        
    # 10 epoch 단위 저장
    if (epoch + 1) % 20 == 1 or (epoch + 1) == epochs:
        save_path = os.path.join(ckpt_dir, f"{tag}_ep{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"[Checkpoint] Saved periodic checkpoint: {save_path}")

wandb.finish()