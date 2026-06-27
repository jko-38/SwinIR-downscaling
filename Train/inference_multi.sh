#!/usr/bin/env bash
set -euo pipefail

# ===== 사용자 설정 =====
RES=0.2
CKPT_ROOT="./checkpoints/${RES}deg"
GPUS=(0 1)           # 사용할 GPU 리스트
MAX_JOBS=2           # 동시에 돌릴 작업 수 (보통 = ${#GPUS[@]})
KFOLDS=(0 1 2 3 4 5)   # 돌릴 k-fold 목록 (예: 0~5 )
PY="python"
CFG="./config.yaml"
INFER="inference.py"
# ======================

running=0
gpu_idx=0

for kfold in "${KFOLDS[@]}"; do
  CKPT_DIR="${CKPT_ROOT}"

  # 패턴: *_${kfold}_residual_f_0.2deg_*_latest.pt
  # (glob 미매칭 시 literal 남는 문제 방지 위해 find 사용)
  #mapfile -t CKPTS < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name "*_${kfold}_${RES}deg_residual_f_latest.pt" | sort)
  # mapfile -t CKPTS < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name "*_${kfold}_${RES}deg_residual_f_topoX_latest.pt" | sort)
  mapfile -t CKPTS < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name "SwinIR_${kfold}_${RES}deg_residual_f_ldaps_latest.pt" | sort)

  if (( ${#CKPTS[@]} == 0 )); then
    echo "[INFO] No checkpoints for kfold=${kfold} in ${CKPT_DIR}"
    continue
  fi

  for CKPT in "${CKPTS[@]}"; do
    F="$(basename "$CKPT")"

    # 모델 매핑
    if   [[ "$F" == edsr_"${kfold}"_* ]];  then M="edsr"
    elif [[ "$F" == srcnn_"${kfold}"_* ]]; then M="srcnn"
    elif [[ "$F" == srgan_"${kfold}"_* ]]; then M="srgan"
    elif [[ "$F" == SwinIR_"${kfold}"_* ]]; then M="swinir"
    else
      echo "[SKIP][k=${kfold}] Unrecognized model in filename: $F"
      continue
    fi

    GPU=${GPUS[$gpu_idx]}
    gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))

    echo "[RUN][k=${kfold}][GPU=${GPU}] $F (model=${M})"
    "$PY" "$INFER" --config "$CFG" --ckpt "$CKPT" --model "$M" --gpu "$GPU" --res "$RES" --k_fold "$kfold" &

    running=$((running + 1))
    if (( running >= MAX_JOBS )); then
      # 동시 작업 수 도달 시 하나 끝날 때까지 대기
      wait -n
      running=$((running - 1))
    fi
  done
done

# 남은 작업 기다리기
wait
echo "✅ All done."
