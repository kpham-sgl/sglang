#!/usr/bin/env bash
# One-shot devbox setup for the PD decode-radix + EAGLE3 repro.
set -euo pipefail
export UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1 PIP_BREAK_SYSTEM_PACKAGES=1
mkdir -p /scratch/logs /scratch/repro-out
cd /scratch
if [ ! -d sglang ]; then
  git clone -q --depth 50 -b pd-decode-radix-eagle3-repro https://github.com/kpham-sgl/sglang.git
fi
cd sglang && git log --oneline -1
SGLANG_RUST_BUILD_MODE=never SGLANG_BUILD_RUST_EXTS=none python3 -m pip install -q -e python --no-deps 2>&1 | tail -2 || true
python3 -c "import sglang, os; print('sglang from', os.path.dirname(sglang.__file__))"
python3 -c "import nixl._api; print('nixl OK')" || echo "nixl MISSING"
python3 -c "import mooncake.engine; print('mooncake OK')" || echo "mooncake MISSING"
python3 -c "import sglang_router; print('router OK')" || echo "router MISSING"
echo "=== models in shared cache"
ls -d /cluster-storage/models/hub/models--openai--gpt-oss-120b /cluster-storage/models/hub/models--lmsys--EAGLE3-gpt-oss-120b-bf16 /cluster-storage/models/hub/models--openai--gpt-oss-20b 2>&1
echo "=== rdma"
(ibdev2netdev 2>/dev/null || ls /sys/class/infiniband 2>/dev/null || echo "no ib devices") | head
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
