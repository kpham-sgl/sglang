#!/usr/bin/env bash
# colocated (non-PD) gpt-oss-20b + EAGLE3 + page 64 + chunked prefill; usage: coloc.sh <TAG> [extra server args]
TAG=$1; shift; D=$(dirname "$(readlink -f "$0")"); source "$D/env.sh"
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=1 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1
CUDA_VISIBLE_DEVICES=${COLOC_GPU:-0} setsid nohup sglang serve --model-path openai/gpt-oss-20b --trust-remote-code --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --page-size 64 --mem-fraction-static 0.6 --max-total-tokens 65536 --max-running-requests 16 --context-length 8192 \
  --chunked-prefill-size 256 --max-prefill-tokens 256 --attention-backend triton --grammar-backend none --sampling-backend pytorch --skip-server-warmup \
  --speculative-algorithm EAGLE3 --speculative-draft-model-path zhuyksir/EAGLE3-gpt-oss-20b-bf16 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --host 127.0.0.1 --port 30000 "$@" > /scratch/logs/$TAG.log 2>&1 &
for i in $(seq 1 60); do curl -sf http://127.0.0.1:30000/health >/dev/null && break; sleep 5; done; echo "coloc up after ~$((i*5))s"
HF_HUB_OFFLINE=0 python3 - <<PY 2>&1 | grep -v Warning | tail -2
from sglang.test.kits.cache_hit_kit import run_multiturn_cache_hit_test
r = run_multiturn_cache_hit_test(base_url="http://127.0.0.1:30000", model_path="openai/gpt-oss-20b", num_clients=4, num_rounds=2, request_length=384, output_length=64, max_parallel=4)
print("multiturn:", r["overall"])
PY
sleep 10
echo "=== REPOINT"; grep -a "REPOINT" /scratch/logs/$TAG.log | head -3 | cut -c1-400
echo "=== leaks"; grep -a -m2 "Mem Leak Detected\|memory leak detected" /scratch/logs/$TAG.log | cut -c1-300
echo "=== alive:"; curl -sf http://127.0.0.1:30000/health >/dev/null && echo yes || echo NO
