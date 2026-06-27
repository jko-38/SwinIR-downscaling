#model_swinir_v3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class positional_embed(nn.Module):
    def __init__(self, harmonics: int = 1):
        super().__init__()
        assert harmonics >= 1
        self.harmonics = harmonics
    @property
    def out_channels(self):  # sin,cos for lat & lon, per harmonic
        return 4 * self.harmonics
    def forward(self, latlon):  # (B,2,H,W) = [lat_rad, lon_rad]
        lat = latlon[:, 0:1]; lon = latlon[:, 1:2]
        feats = []
        for k in range(1, self.harmonics+1):
            kf = float(k)
            feats += [torch.sin(kf*lat), torch.cos(kf*lat),
                    torch.sin(kf*lon), torch.cos(kf*lon)]
        return torch.cat(feats, dim=1)  # (B,4K,H,W)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def build_rel_pos(window_size, num_heads):
    Ws = window_size
    coords_h, coords_w = torch.arange(Ws), torch.arange(Ws)
    try:
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
    except TypeError:
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
    coords_flat = coords.view(2, -1)
    rel = coords_flat[:, :, None] - coords_flat[:, None, :]
    rel = rel.permute(1, 2, 0).contiguous()
    rel[:, :, 0] += Ws - 1
    rel[:, :, 1] += Ws - 1
    rel_index = rel[:, :, 0] * (2 * Ws - 1) + rel[:, :, 1]
    bias_table = nn.Parameter(torch.zeros(((2 * Ws - 1)**2, num_heads)))
    nn.init.trunc_normal_(bias_table, std=0.02)
    return bias_table, rel_index

def compute_attn_mask(H, W, window_size, shift_size, device):
    if shift_size == 0:
        return None
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    Hp, Wp = H + pad_h, W + pad_w
    img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
    slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    cnt = 0
    for h in slices:
        for w in slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size**2)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        table, index = build_rel_pos(window_size, num_heads)
        self.rel_pos_bias_table = table
        self.register_buffer("rel_pos_index", index, persistent=False)

    def forward(self, x, mask=None):
        Bn, N, C = x.shape
        qkv = self.qkv(x).reshape(Bn, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scale = (q.shape[-1])**-0.5
        attn = (q * scale) @ k.transpose(-2, -1)
        bias = self.rel_pos_bias_table[self.rel_pos_index.view(-1)].view(N, N, self.num_heads).permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bn // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        return self.proj(out)

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.ws, self.shift = window_size, shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x, H, W, attn_mask=None):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        pad_h, pad_w = (self.ws - H % self.ws) % self.ws, (self.ws - W % self.ws) % self.ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
        Hp, Wp = H + pad_h, W + pad_w
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        windows = window_partition(x, self.ws)
        out = self.attn(windows, mask=attn_mask)
        x = window_reverse(out, self.ws, Hp, Wp)
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, L, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class RSTB(nn.Module): # Residual Swin Transformer Block
    def __init__(self, dim, depth, num_heads, window_size=8):
        super().__init__()
        self.blocks = nn.ModuleList([SwinBlock(dim, num_heads, window_size, 0 if i % 2 == 0 else window_size // 2) for i in range(depth)])
        self.window_size = window_size
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, H, W):
        shortcut = x
        attn_mask = compute_attn_mask(H, W, self.window_size, self.window_size // 2, x.device)
        for blk in self.blocks:
            x = blk(x, H, W, attn_mask=attn_mask)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x + shortcut

# =========================================================
# 💡  업샘플링 모듈
# =========================================================

class SingleUpsampleBlock(nn.Module):
    """'single' 모드용: 기존 PixelShuffle 기반 2x 업샘플링 블록"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.GELU()
        )
    def forward(self, x):
        return self.block(x)

class DualUpsampleBlock(nn.Module):
    """'dual' 모드용: 기존 PixelShuffle 기반 2x 업샘플링 + Interpolation 업샘플링 블록"""
    def __init__(self, dim):
        super().__init__()
        # 1. PixelShuffle 브랜치
        self.conv_ps_expand = nn.Conv2d(dim, dim * 4, 3, 1, 1) 
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.post_conv_ps = nn.Conv2d(dim, dim, 3, 1, 1) # PixelShuffle 후 정제 Conv
        
        # 2. Interpolation 브랜치 (Bilinear + Conv)
        self.pre_conv_interp = nn.Conv2d(dim, dim, 3, 1, 1) 
        self.post_conv_interp = nn.Conv2d(dim, dim, 3, 1, 1) 
        
        # 3. 융합 (Fusion) 레이어
        self.fusion = nn.Sequential(
            nn.Conv2d(dim + dim, dim, 3, 1, 1), 
            nn.GELU()
        )
        
    def forward(self, x):
        # 브랜치 1: PixelShuffle
        x_ps = self.pixel_shuffle(self.conv_ps_expand(x))
        x_ps = self.post_conv_ps(x_ps) 
        
        # 브랜치 2: Interpolation
        x_interp = self.pre_conv_interp(x) 
        x_interp = F.interpolate(x_interp, scale_factor=2, mode='bilinear', align_corners=False) 
        x_interp = self.post_conv_interp(x_interp)
        
        x_cat = torch.cat([x_ps, x_interp], dim=1)
        x_out = self.fusion(x_cat)
        return x_out
    
class ProgressiveUpsample(nn.Module):
    """
    'mode'에 따라 'single' 또는 'dual' 업샘플링 블록을 쌓는 모듈
    'single' 모드일 때만 최종 refine Conv 레이어를 적용
    """
    def __init__(self, dim, up_steps, mode): 
        super().__init__()
        self.mode = mode  # 👈 모드 저장
        mods = []
        
        if mode == 'single':
            Block = SingleUpsampleBlock
        elif mode == 'dual':
            Block = DualUpsampleBlock
        else:
            raise ValueError(f"Unknown upsample mode: {mode}. Must be 'single' or 'dual'.")
        
        for _ in range(up_steps):
            mods.append(Block(dim))
            
        self.body = nn.Sequential(*mods)
        
        # 'dual' 모드는 마지막 블록에 fusion(refine)이 포함되어 있으므로 None
        if self.mode == 'single':
            self.refine = nn.Conv2d(dim, dim, 3, 1, 1)
        else:
            self.refine = None  

    def forward(self, x):
        x = self.body(x)
        if self.refine is not None:
            x = self.refine(x)
        return x

# =========================================================
# 💡 최종 모델: SwinIR_v3 (v3)
# =========================================================
class SwinIR_v3(nn.Module):
    def __init__(self,
                in_channels_dyn=4,    # T2m, U10, V10, SP
                static_channels=5,    # Elevation, Slope, lat, lon, Mask
                embed_dim=96,
                num_rstb=4,
                depths=(4, 4, 4, 4),
                num_heads=(4, 4, 4, 4),
                window_size=8,
                up_steps=4,           # 16x upsampling
                upsample_mode=None,
                target_size=(731, 430)):
        super().__init__()
        self.target_size = target_size
        self.upsample_mode = upsample_mode

        # 0) LR용 positional embedding (sin/cos)
        self.pe_lr = positional_embed(harmonics=1)
        pe_lr_ch = self.pe_lr.out_channels  # 4*harmonics

        # --- 1. 정적(지형) 인코더 ---
        self.static_encoder = nn.Sequential(
            nn.Conv2d(static_channels, embed_dim // 2, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 1, 1),)

        # --- 2. 얕은 특징 추출기 (Shallow Feature Extractor) ---
        # X_dyn(4) + LAI(1) + LatLon positional embedding(4) = 9 채널 입력
        self.embed = nn.Conv2d(in_channels_dyn + 1 + pe_lr_ch, embed_dim, 3, 1, 1)

        # --- 3. SwinIR 백본 (Deep Feature Extractor) ---
        self.rstb_stack = nn.ModuleList([
            RSTB(dim=embed_dim, depth=d, num_heads=h, window_size=window_size)
            for d, h in zip(depths[:num_rstb], num_heads[:num_rstb])])
        self.backbone_conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # --- 4. Upsampling (Reconstruction) ---
        self.upsample = ProgressiveUpsample(embed_dim, up_steps=up_steps, mode=self.upsample_mode)
        self.head_conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # --- 5. 최종 융합 디코더 ---
        # dyn_feat(96) + static_feat(96) 
        self.final_fusion = nn.Sequential(
            nn.Conv2d(embed_dim + embed_dim, embed_dim, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(embed_dim, embed_dim // 2, 3, 1, 1),  nn.ReLU(True),
            nn.Conv2d(embed_dim // 2, 1, 3, 1, 1) # 최종 출력: 1채널 (온도 잔차)
        )
        
    # ---- 유틸: 정적/좌표/LAI 정렬 ----
    @staticmethod
    def _ensure_bchw(x):
        if x.ndim == 2:   x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3: x = x.unsqueeze(0)
        elif x.ndim == 4: pass
        else: raise ValueError(f"Unsupported ndim={x.ndim}")
        return x
    @staticmethod
    def _match_hw(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x
    @staticmethod
    def _expand_B(x, B):
        if x.shape[0] == 1 and B > 1:
            x = x.expand(B, -1, -1, -1)
        return x
    def _prepare_lai(self, lai, x_dyn):
        B = x_dyn.shape[0]
        lai = self._ensure_bchw(lai).to(x_dyn.device)   # -> (1?,1,H,W)
        if lai.shape[1] != 1: raise ValueError("LAI must be 1 channel")
        lai = self._match_hw(lai, x_dyn)
        return self._expand_B(lai, B)
    def _prepare_latlon_lr(self, latlon_lr, x_dyn):
        B = x_dyn.shape[0]
        latlon_lr = self._ensure_bchw(latlon_lr).to(x_dyn.device)  # -> (1?,2,H_lr,W_lr)
        if latlon_lr.shape[1] != 2: raise ValueError("latlon_lr must be 2 channels [lat,lon]")
        latlon_lr = self._match_hw(latlon_lr, x_dyn)
        return self._expand_B(latlon_lr, B)
    
    def _prepare_static_hr(self, x_static, B):
        """
        x_static dict에서 HR 정적 채널 5개를 (B,5,H_hr,W_hr)로 스택:
        [elevation(1), slope(1), lat(1), lon(1), mask(1)]
        """
        device = next(self.parameters()).device
        def _to_1ch(x):
            x = self._ensure_bchw(x.to(device).float())  # (1,1,H,W) or (1?,H,W)->(1,1,H,W)
            if x.shape[1] != 1:
                raise ValueError("Static single-channel tensor must have 1 channel.")
            return x

        # 필수 키 점검
        required = ["elevation", "slope", "mask", "latlon_HR"]
        for k in required:
            if k not in x_static:
                raise KeyError(f"Missing static key: '{k}' in x_static")

        elev = _to_1ch(x_static["elevation"])
        slope = _to_1ch(x_static["slope"])
        mask = _to_1ch(x_static["mask"])

        # latlon_HR 은 (2,H,W) 또는 (1,2,H,W) 여야 함
        latlon_hr = x_static["latlon_HR"].to(device).float()
        latlon_hr = self._ensure_bchw(latlon_hr)         # -> (1?,C,H,W)
        if latlon_hr.shape[1] != 2:
            raise ValueError("latlon_HR must have 2 channels [lat, lon].")

        # 공간크기 정렬 (모두 HR 동일크기로 맞춤)
        H_hr, W_hr = elev.shape[-2], elev.shape[-1]
        slope = self._match_hw(slope, elev)
        mask  = self._match_hw(mask, elev)
        latlon_hr = self._match_hw(latlon_hr, elev)      # (1,2,H_hr,W_hr)

        # (1,5,H,W) 스택 후 배치 확장
        static_hr = torch.cat([elev, slope, latlon_hr, mask], dim=1)  # (1,5,H_hr,W_hr)
        static_hr = self._expand_B(static_hr, B)                      # (B,5,H_hr,W_hr)
        return static_hr
    
    def forward(self, x_dyn, x_static):
        B, _, H_lr, W_lr = x_dyn.shape
        device = x_dyn.device

        lai       = x_static['lai'].to(device).float()           # (1,H_lr,W_lr) or (H_lr,W_lr)
        latlon_lr = x_static['latlon_LR'].to(device).float()     # (2,H_lr,W_lr)
        
        lai = self._prepare_lai(lai, x_dyn)                 # (B,1,H_lr,W_lr)
        latlon_lr = self._prepare_latlon_lr(latlon_lr, x_dyn)  # (B,2,H_lr,W_lr)
        
        # --- 단계 1: 정적 특징 준비 (HR -> LR) ---
        static_hr = self._prepare_static_hr(x_static, B)
        static_feat_hr = self.static_encoder(static_hr)
        static_feat_lr = F.interpolate(static_feat_hr, size=(H_lr, W_lr), mode='bilinear', align_corners=False)

        # --- 단계 2: 얕은 특징 추출 및 LR 융합 ---

        pe_lr = self.pe_lr(latlon_lr)                       # (B,4K,H_lr,W_lr)

        x_input = torch.cat([x_dyn, lai, pe_lr], dim=1)
        feat_shallow = self.embed(x_input)
        feat_fused_lr = feat_shallow + static_feat_lr

        # --- 단계 3: 깊은 특징 추출 (SwinIR Backbone) ---
        feat_deep_token = rearrange(feat_fused_lr, 'b c h w -> b (h w) c')
        for rstb in self.rstb_stack:
            feat_deep_token = rstb(feat_deep_token, H_lr, W_lr)
        feat_deep = rearrange(feat_deep_token, 'b (h w) c -> b c h w', h=H_lr, w=W_lr)
        feat_deep = self.backbone_conv(feat_deep)

        # --- 단계 4: 전역 잔차 연결 및 고해상도 복원  ---
        feat_up = self.upsample(feat_deep + feat_shallow)
        feat_up = self.head_conv(feat_up)
        if feat_up.shape[-2:] != self.target_size:
            feat_up = F.interpolate(feat_up, size=self.target_size, mode='bilinear', align_corners=False)

        # --- 단계 5: 최종 융합 (HR resolution) ---
        final_input = torch.cat([feat_up, static_feat_hr], dim=1)
        out = self.final_fusion(final_input)

        return out
    
    
    
# =========================================================
# 💡 Ablation 모델: SwinIR_v3_wo_topo (지형, 좌표, LAI 등 모든 정적 변수 제거)
# =========================================================
class SwinIR_v3_wo_topo(nn.Module):
    def __init__(self,
                 in_channels_dyn=4,    # T2m, U10, V10, SP (순수 동적 변수만)
                 embed_dim=96,
                 num_rstb=4,
                 depths=(4, 4, 4, 4),
                 num_heads=(4, 4, 4, 4),
                 window_size=8,
                 up_steps=4,           # 16x upsampling
                 upsample_mode=None,
                 target_size=(731, 430)):
        super().__init__()
        self.target_size = target_size
        self.upsample_mode = upsample_mode

        # --- 2. Shallow Feature Extractor ---
        self.embed = nn.Conv2d(in_channels_dyn, embed_dim, 3, 1, 1)

        # --- 3. SwinIR backbone (Deep Feature Extractor) ---
        self.rstb_stack = nn.ModuleList([
            RSTB(dim=embed_dim, depth=d, num_heads=h, window_size=window_size)
            for d, h in zip(depths[:num_rstb], num_heads[:num_rstb])])
        self.backbone_conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # --- 4. Upsampling (Reconstruction) ---
        self.upsample = ProgressiveUpsample(embed_dim, up_steps=up_steps, mode=self.upsample_mode)
        self.head_conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # --- 5. 최종 융합 디코더 ---
        self.final_fusion = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(embed_dim, embed_dim // 2, 3, 1, 1),  nn.ReLU(True),
            nn.Conv2d(embed_dim // 2, 1, 3, 1, 1) # 최종 출력: 1채널 (온도 잔차)
        )

    def forward(self, x_dyn, x_static):
        B, _, H_lr, W_lr = x_dyn.shape
        
        # --- 단계 2: 얕은 특징 추출 ---
        feat_shallow = self.embed(x_dyn)

        # --- 단계 3: 깊은 특징 추출 (SwinIR Backbone) ---
        feat_deep_token = rearrange(feat_shallow, 'b c h w -> b (h w) c')
        for rstb in self.rstb_stack:
            feat_deep_token = rstb(feat_deep_token, H_lr, W_lr)
        feat_deep = rearrange(feat_deep_token, 'b (h w) c -> b c h w', h=H_lr, w=W_lr)
        feat_deep = self.backbone_conv(feat_deep)

        # --- 단계 4: 전역 잔차 연결 및 고해상도 복원  ---
        feat_up = self.upsample(feat_deep + feat_shallow)
        feat_up = self.head_conv(feat_up)
        if feat_up.shape[-2:] != self.target_size:
            feat_up = F.interpolate(feat_up, size=self.target_size, mode='bilinear', align_corners=False)

        # --- 단계 5: 최종 융합 (HR resolution) ---
        out = self.final_fusion(feat_up)
        return out
