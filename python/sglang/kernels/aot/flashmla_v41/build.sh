#!/bin/bash
# Builds flashmla_v41_ops (torch.ops.sgl_flashmla_v41) without the rest of sgl-kernel and installs it
# where sglang.kernels.ops.attention.dsv4.flashmla_v41 loads it from.
#   FLASHMLA_V41_SOURCE_DIR  FlashMLA 2e5429f checkout with csrc/3rdparty/cutlass (fetched when unset)
#   CUDA_HOME                a CUDA >= 13.1 toolkit (13.2 for the native bf16x2 conversions)
#   BUILD_DIR                default /tmp/flashmla_v41_build; JOBS default nproc
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/../../ops/attention/dsv4/_flashmla_v41"
BUILD_DIR="${BUILD_DIR:-/tmp/flashmla_v41_build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
args=(-S "$HERE" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME"
      -DPython3_EXECUTABLE="$(command -v python3)")
[ -n "${FLASHMLA_V41_SOURCE_DIR:-}" ] && args+=(-DFLASHMLA_V41_SOURCE_DIR="$FLASHMLA_V41_SOURCE_DIR")
cmake "${args[@]}"
cmake --build "$BUILD_DIR" -j "${JOBS:-$(nproc)}"
mkdir -p "$DEST"
cp "$BUILD_DIR/libflashmla_v41_ops.so" "$DEST/"
echo "installed $DEST/libflashmla_v41_ops.so"
