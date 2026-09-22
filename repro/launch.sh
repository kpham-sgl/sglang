#!/usr/bin/env bash
# usage: launch.sh <prefill|decode|lb> [A|B]   (A = faithful, B = localize)
set -uo pipefail
ROLE=$1; MODE=${2:-A}
source "$(dirname "$0")/env.sh"
SPEC=(--speculative-algorithm EAGLE3 --speculative-draft-model-path "$DRAFT"
      --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
[ "${NO_SPEC:-0}" = 1 ] && SPEC=()
COMMON=(--model-path "$MODEL" --trust-remote-code --dtype bfloat16 --kv-cache-dtype bfloat16
  --page-size "${PAGE_SIZE:-64}" --mem-fraction-static "${MEM_FRAC:-0.6}"
  --max-total-tokens "${MAX_TOTAL_TOKENS:-65536}" --max-running-requests 16
  --context-length 8192 --chunked-prefill-size 4096 --max-prefill-tokens 4096
  --attention-backend "$ATTN" --grammar-backend none --sampling-backend pytorch --skip-server-warmup
  --reasoning-parser gpt-oss --served-model-name gpt-oss-120b
  --disaggregation-transfer-backend "$TRANSFER" --host 127.0.0.1 "${SPEC[@]}" ${EXTRA_ARGS:-})
case $ROLE in
  prefill)
    CUDA_VISIBLE_DEVICES=$PREFILL_GPUS exec sglang serve "${COMMON[@]}" --tp-size "$PREFILL_TP" --disaggregation-mode prefill \
      --disaggregation-bootstrap-port $BOOTSTRAP_PORT --port $PREFILL_PORT --nccl-port $NCCL_P ;;
  decode)
    export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1
    if [ "$MODE" = B ]; then
      export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=1 SGLANG_DEBUG_MEMORY_POOL=1 \
             SGLANG_CHECK_KV_PAGE_INVARIANTS=1 SGLANG_ENABLE_TREE_CACHE_SANITY_CHECK=1
    fi
    RADIX=(--disaggregation-decode-enable-radix-cache); [ "${NO_RADIX:-0}" = 1 ] && RADIX=()
    CUDA_VISIBLE_DEVICES=$DECODE_GPUS exec sglang serve "${COMMON[@]}" --tp-size "$DECODE_TP" --disaggregation-mode decode \
      --disaggregation-bootstrap-port $BOOTSTRAP_PORT --port $DECODE_PORT --nccl-port $NCCL_D "${RADIX[@]}" ;;
  lb)
    exec python3 -m sglang_router.launch_router --pd-disaggregation --mini-lb \
      --prefill http://127.0.0.1:$PREFILL_PORT --decode http://127.0.0.1:$DECODE_PORT \
      --host 127.0.0.1 --port $LB_PORT ;;
esac
