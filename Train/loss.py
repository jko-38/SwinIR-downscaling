from pytorch_msssim import ssim
import torch
from torch import nn, optim
import yaml
import argparse
import torch.nn.functional as F

with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# def ssim_loss(pred, target):
#     """
#     SSIM-based loss: SSIM(pred, target)
#     """
#     return ssim(pred, target, data_range=1.0, size_average=True)

def ssim_loss(pred, target):
    """
    SSIM-based loss 계산을 위한 함수.
    train.py에서 (1 - ssim) 처리를 하므로, 여기서는 SSIM 값 자체를 리턴합니다.
    중요: Z-score Scaling 데이터의 범위를 반영하기 위해 data_range를 동적으로 계산합니다.
    """
    # 배치 내 최대-최소 차이를 계산하여 data_range로 사용
    # .item()을 사용하여 그래프 연결을 끊고 상수로 취급 (학습 안정성)
    max_val = max(pred.max().item(), target.max().item())
    min_val = min(pred.min().item(), target.min().item())
    dynamic_range = max_val - min_val
    
    # 혹시 모를 0 나누기 방지 및 최소 범위 보장
    dynamic_range = max(dynamic_range, 1e-7)

    # SSIM 계산
    ssim_val = ssim(pred, target, data_range=dynamic_range, size_average=True)
    return ssim_val


def old_grad_loss(pred, target):
    def gradient(tensor):
        dx = tensor[..., :-1, 1:] - tensor[..., :-1, :-1]
        dy = tensor[..., 1:, :-1] - tensor[..., :-1, :-1]
        return dx, dy
    dx_pred, dy_pred = gradient(pred)
    dx_target, dy_target = gradient(target)
    return ((dx_pred - dx_target) ** 2 + (dy_pred - dy_target) ** 2).mean()

def grad_loss(pred, target, mask=None, p=1):
    """
    Gradient loss comparing finite differences along x/y.
    - pred, target: (B,C,H,W) or (B,1,H,W). (H,W)도 허용(자동 확장)
    - mask: None or (H,W) or (B,1,H,W) or (B,H,W); 0/1 값. (유효영역=1)
    - p: 1 -> L1, 2 -> L2(squared)
    반환: scalar (torch.Tensor)
    """
    # --- shape 정리 ---
    # pred/target이 (H,W)일 수도 있으니 강제 4D로
    if pred.dim() == 2:         # (H,W)
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif pred.dim() == 3:       # (B,H,W)
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)

    assert pred.shape == target.shape, f"shape mismatch: {pred.shape} vs {target.shape}"
    B, C, H, W = pred.shape

    # --- mask 정리 (torch로, dtype/device 맞춤) ---
    if mask is not None:
        if isinstance(mask, torch.Tensor) is False:
            mask = torch.as_tensor(mask)
        mask = mask.to(device=pred.device, dtype=pred.dtype)

        # 허용 shape: (H,W), (B,H,W), (B,1,H,W)
        if mask.dim() == 2:             # (H,W)
            mask = mask.unsqueeze(0).unsqueeze(0)      # (1,1,H,W)
            mask = mask.expand(B, 1, H, W)             # 배치 브로드캐스트
        elif mask.dim() == 3:           # (B,H,W)
            mask = mask.unsqueeze(1)                     # (B,1,H,W)
        elif mask.dim() == 4:           # (B,1,H,W) or (B,C,H,W)
            if mask.size(1) != 1:
                # 채널 마스크면 채널 평균으로 1채널 마스크로 축소
                mask = mask.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")

    # --- finite differences (방향별) ---
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]   # (B,C,H,W-1)
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]   # (B,C,H-1,W)
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_t = target[..., 1:, :] - target[..., :-1, :]

    if p == 1:
        lx = (dx_p - dx_t).abs()
        ly = (dy_p - dy_t).abs()
    else:  # p == 2 (squared)
        lx = (dx_p - dx_t) ** 2
        ly = (dy_p - dy_t) ** 2

    if mask is not None:
        # 엣지 유효 페어 마스크 (양쪽 픽셀이 모두 1일 때만 유효)
        mask_x = mask[..., :, 1:] * mask[..., :, :-1]     # (B,1,H,W-1)
        mask_y = mask[..., 1:, :] * mask[..., :-1, :]     # (B,1,H-1,W)

        # 채널 차원 브로드캐스트
        if mask_x.size(1) == 1 and C > 1:
            mask_x = mask_x.expand(B, C, mask_x.size(-2), mask_x.size(-1))
            mask_y = mask_y.expand(B, C, mask_y.size(-2), mask_y.size(-1))

        # 방향별로 먼저 스칼라로 축약
        num_x = torch.clamp_min(mask_x.sum(), 1.0)
        num_y = torch.clamp_min(mask_y.sum(), 1.0)
        loss_x = (lx * mask_x).sum() / num_x
        loss_y = (ly * mask_y).sum() / num_y

        # 유효 페어 수로 가중 결합(혹은 단순 평균도 가능)
        loss = (loss_x * num_x + loss_y * num_y) / (num_x + num_y)
        return loss
    else:
        # 마스크 없으면 방향별 평균 후 평균
        loss_x = lx.mean()
        loss_y = ly.mean()
        return 0.5 * (loss_x + loss_y)
# def generator_loss(fake, real, fake_logits, ssim_lambda=0.3, lambda_adv=0.001, criterion_type='mae'):
#     mae_loss = nn.L1Loss()
#     bce_loss = nn.BCEWithLogitsLoss()
#     mae = mae_loss(fake, real)
#     adv = bce_loss(fake_logits, torch.ones_like(fake_logits))
#     if 'mae+ssim' in criterion_type:
#         ssim = ssim_loss(fake, real)
#         return mae + ssim_lambda * (1 - ssim) + lambda_adv * adv
#     else:
#         return mae + lambda_adv * adv

# def discriminator_loss(real_logits, fake_logits):
#     bce_loss = nn.BCEWithLogitsLoss()
#     real_loss = bce_loss(real_logits, torch.ones_like(real_logits))
#     fake_loss = bce_loss(fake_logits, torch.zeros_like(fake_logits))
#     return (real_loss + fake_loss) / 2


def generator_loss(
    fake, real, fake_logits,
    adv_type='hinge',              # 'bce' 또는 'hinge'
    criterion_type='mae',        # 'mae', 'mae+ssim', 'mae+grad', 'mae+ssim+grad'
    ssim_lambda=0.3,
    grad_lambda=0.0,
    lambda_adv=0.001,
    bce_label_smooth=0.9
):
    # 1) Content Loss
    loss = F.l1_loss(fake, real)  # MAE
    if 'mae+ssim' in criterion_type:
        ssim_val = ssim_loss(fake, real)
        loss = loss + ssim_lambda * (1.0 - ssim_val)
    elif 'mae+grad' in criterion_type and grad_lambda > 0.0:
        loss = loss + grad_lambda * grad_loss(fake, real)
    elif 'mae+grad+ssim' in criterion_type and grad_lambda > 0.0:
        ssim_val = ssim_loss(fake, real)
        loss = loss + grad_lambda * grad_loss(fake, real) + ssim_lambda * (1.0 - ssim_val)

    # 2) Adversarial Loss (항상 포함)
    if adv_type == 'bce':
        bce = nn.BCEWithLogitsLoss()
        real_label = torch.full_like(fake_logits, fill_value=bce_label_smooth)
        adv = bce(fake_logits, real_label)
    elif adv_type == 'hinge':
        adv = - fake_logits.mean()
    else:
        raise ValueError("adv_type must be 'bce' or 'hinge'")

    return loss + lambda_adv * adv

# ---------------------------
# Discriminator Loss
# ---------------------------
def discriminator_loss(
    real_logits, fake_logits,
    adv_type='hinge',
    bce_label_smooth=0.9
):
    if adv_type == 'bce':
        bce = nn.BCEWithLogitsLoss()
        real_t = torch.full_like(real_logits, fill_value=bce_label_smooth)
        fake_t = torch.zeros_like(fake_logits)
        return 0.5 * (bce(real_logits, real_t) + bce(fake_logits, fake_t))
    elif adv_type == 'hinge':
        return torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()
    else:
        raise ValueError("adv_type must be 'bce' or 'hinge'")