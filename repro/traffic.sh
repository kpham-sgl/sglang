#!/usr/bin/env bash
# usage: traffic.sh <TAG> [steps: e.g. "1 2 3"]; checks decode liveness after each idle window
set -uo pipefail
TAG=$1; STEPS=${2:-"1 2 3"}; D=$(dirname "$(readlink -f "$0")")
source "$D/env.sh"
alive() { pgrep -f "disaggregation-mode decode .*--port $DECODE_PORT" >/dev/null && curl -sf http://127.0.0.1:$DECODE_PORT/health >/dev/null; }
report() { echo "[$(date -u +%T)] $1 decode_alive=$(alive && echo yes || echo NO)"; grep -c "memory leak detected\|Mem Leak Detected\|AssertionError" /scratch/logs/$TAG.decode.log | sed 's/^/  leak_lines=/'; }
for s in $STEPS; do
  case $s in
    1) echo "[$(date -u +%T)] step1 multiturn kit + gsm8k100"
       HF_HUB_OFFLINE=0 python3 - <<PY
from sglang.test.kits.cache_hit_kit import run_multiturn_cache_hit_test
r = run_multiturn_cache_hit_test(base_url="http://127.0.0.1:$LB_PORT", model_path="$MODEL", num_clients=8, num_rounds=2, request_length=384, output_length=64, max_parallel=4)
print("multiturn:", r["overall"])
PY
       python3 -m sglang.test.run_eval --base-url http://127.0.0.1:$LB_PORT --eval-name gsm8k --num-examples 100 --num-threads 8 --max-tokens 512 2>&1 | tail -3
       sleep 60; report "after step1 (60s idle)" ;;
    2) echo "[$(date -u +%T)] step2 gsm8k500 fill pool"
       python3 -m sglang.test.run_eval --base-url http://127.0.0.1:$LB_PORT --eval-name gsm8k --num-examples 500 --num-threads 16 --max-tokens 1024 --temperature 0.0 2>&1 | tail -3
       sleep 30; report "after step2 (30s idle)" ;;
    3) echo "[$(date -u +%T)] step3 hit-vs-cold"
       LB_PORT=$LB_PORT DECODE_PORT=$DECODE_PORT python3 "$D/hit_vs_cold.py" 12 /scratch/repro-out/$TAG.hit-vs-cold.json 2>&1 | tail -4
       sleep 20; report "after step3 (20s idle)" ;;
  esac
  alive || { echo "decode is DOWN after step $s"; grep -n -A12 "memory leak detected\|Mem Leak Detected\|AssertionError\|Traceback" /scratch/logs/$TAG.decode.log | head -80; exit 2; }
done
echo "all steps done, decode alive"
