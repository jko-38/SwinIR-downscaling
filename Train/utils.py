# utils.py
import os

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

    # dynamic input variables
    for var in config['input_vars']:
        if var == 't2m' and use_ldaps_input:
            updated_paths[var] = os.path.join(base_dir, res_str, f"input__LDAPS_tmpr_data_{data_processing}.pt")
        else:
            updated_paths[var] = os.path.join(base_dir, res_str, f"input_{var}_data_{data_processing}.pt")
            
    # static variables
#   if config.get('use_lai_lr', True):
#       updated_paths['lai'] = os.path.join(base_dir, res_str, config['data_paths']['lai'])  # (1,H_lr,W_lr) or (H_lr,W_lr)

    if config['use_static'] == True : 
        for var in config['static_vars']:
            if var == "lai" or var == "latlon_LR":
                updated_paths[var] = os.path.join(base_dir, res_str, config['data_paths'][var])
            else :
                updated_paths[var] = os.path.join(base_dir, config['data_paths'][var])

    # target은 resolution 상관 없이 고정 경로
    updated_paths['target'] = os.path.join(base_dir, f"target_tmpr_data_{data_processing}.pt")
    return updated_paths


def normalize_to_1x1HW(t, device):
    """
    t: (H,W) or (1,H,W) or (C,H,W) or (B,1,H,W)
    반환: (1,1,H,W)
    - C>1 같은 경우는 호출 측에서 채널별로 루프 돌려주세요.
    """
    if t.dim() == 4:                # (B,C,H,W)
        if t.size(0) == 1 and t.size(1) == 1:
            t = t[0, 0]             # -> (H,W)
        else:
            raise ValueError("Expected (B=1,C=1,H,W) for static tensors when passing batch; got {}".format(tuple(t.shape)))
    if t.dim() == 3:                # (C,H,W) or (1,H,W)
        if t.size(0) == 1:
            t = t[0]                # -> (H,W)
        else:
            # 다채널일 땐 호출부에서 채널별로 처리
            return None
    # 이제 (H,W)
    if t.dim() != 2:
        raise ValueError("Static tensor must reduce to (H,W), got {}".format(tuple(t.shape)))
    return t.unsqueeze(0).unsqueeze(0).to(device)   # (1,1,H,W)


# def update_data_paths_by_resolution(config):
#     """
#     Modify input data paths in config based on the resolution value.
#     Automatically switches ERA5 input filenames to match resolution.
#     """
#     res = float(config.get("res"))  
#     res_str = get_resolution_str(res)

#     # 기본 해상도일 경우 변경 없이 반환
#     if res_str is None:
#         return config["data_paths"]

#     updated_paths = config["data_paths"].copy()
#     for var in ["t2m", "u10", "v10", "sp"]:
#         base = config["data_paths"][var]
#         updated_paths[var] = base.replace(".nc", f".{res_str}.nc")

#     return updated_paths

