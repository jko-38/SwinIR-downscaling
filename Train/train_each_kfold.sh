#!/usr/bin/env bash
set -euo pipefail

# ====== 사용자 설정 ======
CONFIG="./config.yaml"
RES_LIST=(1) 
MODEL="swinir"
PY="python"
SCRIPT="train_SwinIR_v3.py"

# 셸 스크립트 인자
version=$1
START_FOLD=${2:? "START_FOLD 필요"}
END_FOLD=${3:? "END_FOLD 필요"}
GPU_LIST=(0)      # (0 1) 등
# ========================

trap 'echo "⛔ 중단됨"; exit 1' INT TERM

N=${#GPU_LIST[@]}      # GPU 개수 (라운드로빈 할당용)

# 🌟 [수정] K-FOLD 루프가 외부 루프가 됨
for K in $(seq "$START_FOLD" "$END_FOLD"); do
    
    # 현재 K-Fold에서 사용할 GPU를 선택 (이 폴드의 모든 해상도에서 동일 GPU 사용)
    GPU=${GPU_LIST[$(( (K-START_FOLD) % N ))]}
    
    echo "======== K-FOLD $K (Using GPU $GPU) 시작 ========"

    # 🌟 [수정] 해상도(RES) 루프가 내부 루프가 됨
    for RES in "${RES_LIST[@]}"; do
        
        echo "  ---- (K=$K) Resolution $RES 처리 시작 ----"
        START_TS=$(date +%s)

        # 🌟 [수정] --res 인자를 $RES (루프 변수)로 설정
        ARGS=( --config "$CONFIG" --gpu "$GPU" --kfold "$K" --model "$MODEL" --version "$version" )
        ARGS+=( --res "$RES" )

        # 동기 실행 (현재 해상도가 끝나야 다음 해상도로 넘어감)
        "$PY" "$SCRIPT" "${ARGS[@]}"

        END_TS=$(date +%s)
        echo "  ✅ (K=$K) Resolution $RES 완료 (소요: $((END_TS-START_TS))s)"
    
    done # --- 해상도(RES) 루프 종료

    echo "🎉 K-FOLD $K 모든 해상도 처리 완료."

done # --- K-FOLD 루프 종료

echo "🏁 모든 K-FOLD (${START_FOLD}~${END_FOLD}) 및 모든 해상도 순차 실행 완료"
