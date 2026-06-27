#!/bin/bash
model=$1
RES=$2
GPU=0
kfold_start=$3
kfold_end=$4

tag='v1_ep100'

echo "Starting Inference Loop"
echo "Model=${model}, res=${RES}, gpu=${GPU}"
echo "K-fold range: ${kfold_start} to ${kfold_end}"
echo "tag='${tag}'"
echo "----------------------------------------"

for kfold in $(seq $kfold_start $kfold_end); do
  #CKPT_PATH="${model}_${kfold}_${RES}deg_residual_f_${tag}.pt"
  CKPT_PATH="swinir_${kfold}_${RES}deg_residual_f_${tag}.pt"
  
  echo ""
  echo "Running K-fold=${kfold}"
  echo "Checkpoint: ${CKPT_PATH}"
  
  python inference.py \
    --config config.yaml \
    --ckpt "./checkpoints/${RES}deg/${CKPT_PATH}" \
    --res ${RES} \
    --gpu ${GPU} \
    --model ${model} \
    --k_fold ${kfold}
  
  echo "Completed K-fold=${kfold}"
done

echo ""
echo "All inference completed!"
