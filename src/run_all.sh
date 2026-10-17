#!/bin/bash
# Runs the remaining grid on a shared, oversubscribed GPU host (other users' jobs come and go).
# Per run, in order of preference:
#   1. single GPU, bf16 weights      (if weights + 6 GB KV fit in free memory)
#   2. single GPU, FP8 weights       (on-the-fly weight quantisation)
#   3. 2 GPUs tensor-parallel, FP8, eager mode (uses the two GPUs with most free memory)
# vLLM's memory fraction is sized to what is actually free; failures are retried.
# Precision / TP actually used is recorded in results/raw/<model>__<tag>.meta.json.
set -u
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
declare -A WEIGHTS_MIB=([qwen25_7b]=15500 [llama31_8b]=16100 [qwen3_4b]=8100)
MARGIN=1500
run() {  # model tag extra-args...
  local m=$1 tag=$2; shift 2
  [ -f results/raw/${m}__${tag}.meta.json ] && return 0
  local w=${WEIGHTS_MIB[$m]}
  {
    mapfile -t G < <(nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits | tr -d ',' | sort -k2 -nr)
    read g1 f1 total <<< "${G[0]}"; read g2 f2 _ <<< "${G[1]}"
    local a1=$((f1 - MARGIN)) a2=$((f2 - MARGIN)) args="" util="" gpus=""
    if   [ $a1 -ge $((w + 6000)) ];     then gpus=$g1; util=$a1; args="--quant="
    elif [ $a1 -ge $((w / 2 + 3500)) ]; then gpus=$g1; util=$a1; args="--quant=fp8"
    elif [ $a2 -ge $((w / 4 + 1000)) ]; then gpus="$g1,$g2"; util=$a2; args="--quant=fp8 --tp 2 --eager"
    else return 1; fi
    util=$(python -c "print(round(min(0.85, $util/$total), 3))")
    echo "$(date) $m $tag attempt gpus=$gpus util=$util $args"
    CUDA_VISIBLE_DEVICES=$gpus python src/run.py --model $m --tag $tag --gpu_util $util $args "$@" \
      > logs/run_${m}_${tag}.log 2>&1 && { echo "$(date) $m $tag DONE"; return 0; }
    echo "$(date) $m $tag failed: $(grep -E 'Error:' logs/run_${m}_${tag}.log | tail -1 | cut -c1-160)"
    return 1
  }
}
# Non-blocking queue: each cycle launch the first pending job that fits in current free memory.
JOBS=("qwen3_4b greedy" "llama31_8b greedy" "qwen25_7b t07s1" "qwen3_4b t07s1" "llama31_8b t07s1")
for cycle in $(seq 1 400); do
  pending=0
  for j in "${JOBS[@]}"; do
    set -- $j
    [ -f results/raw/${1}__${2}.meta.json ] && continue
    pending=1
    if [ "$2" = t07s1 ]; then run $1 $2 --temperature 0.7 --seed 1 && continue 2; else run $1 $2 && continue 2; fi
  done
  [ $pending = 0 ] && break
  echo "$(date) cycle $cycle: nothing fits ($(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ')MiB free)"
  sleep 90
done
echo ALL_DONE
