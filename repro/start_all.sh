#!/usr/bin/env bash
# start prefill+decode+lb detached; usage: start_all.sh <TAG> [A|B]
set -uo pipefail
TAG=$1; MODE=${2:-A}; D=$(dirname "$(readlink -f "$0")")
source "$D/env.sh"
[ "${NO_KILL:-0}" = 1 ] || { pkill -f "sglang serve" 2>/dev/null; pkill -f launch_router 2>/dev/null; sleep 3; }
setsid nohup bash "$D/launch.sh" prefill "$MODE" > /scratch/logs/$TAG.prefill.log 2>&1 &
setsid nohup bash "$D/launch.sh" decode  "$MODE" > /scratch/logs/$TAG.decode.log 2>&1 &
for i in $(seq 1 120); do
  p=$(curl -sf http://127.0.0.1:$PREFILL_PORT/health >/dev/null && echo 1 || echo 0)
  d=$(curl -sf http://127.0.0.1:$DECODE_PORT/health  >/dev/null && echo 1 || echo 0)
  [ "$p$d" = 11 ] && break
  pgrep -f "disaggregation-mode prefill .*--port $PREFILL_PORT" >/dev/null || { echo "prefill died"; tail -30 /scratch/logs/$TAG.prefill.log; exit 1; }
  pgrep -f "disaggregation-mode decode .*--port $DECODE_PORT"  >/dev/null || { echo "decode died";  tail -30 /scratch/logs/$TAG.decode.log;  exit 1; }
  sleep 10
done
echo "engines up after ~$((i*10))s"
setsid nohup bash "$D/launch.sh" lb > /scratch/logs/$TAG.lb.log 2>&1 &
sleep 5
curl -sf http://127.0.0.1:$DECODE_PORT/server_info | python3 -c "import json,sys; d=json.load(sys.stdin); print({k:d.get(k) for k in ('disable_radix_cache','speculative_algorithm','max_total_num_tokens','page_size','attention_backend')})"
grep -m1 "REPRO:" /scratch/logs/$TAG.decode.log || true
grep -m1 "max_total_num_tokens\|KV size\|#token" /scratch/logs/$TAG.decode.log | head -3
