#!/usr/bin/env bash
# Create a self-contained Python environment for model 3 in ./.venv (no Isaac Lab needed).
#
#   bash setup_env.sh          # torch with CUDA 13.0 (NVIDIA GPU, driver >= 580)
#   bash setup_env.sh --cpu    # CPU-only torch (much smaller); then run the script with --device cpu
#
# Afterwards:  .venv/bin/python liquid_transfer_m3.py --mode play --num_envs 32 --max_steps 1800 --seed 7 --checkpoint policy.pt
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
PY_VERSION="3.12"
TORCH_VERSION="2.12.0"

TORCH_INDEX="https://download.pytorch.org/whl/cu130"
if [[ "${1:-}" == "--cpu" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
elif [[ -n "${1:-}" ]]; then
  echo "Unknown option: $1 (use --cpu or nothing)" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  # uv downloads Python 3.12 itself if the system does not have it
  uv venv --python "$PY_VERSION" "$VENV"
  PIP=(uv pip install --python "$VENV/bin/python")
else
  PYTHON="$(command -v "python$PY_VERSION" || true)"
  if [[ -z "$PYTHON" ]]; then
    echo "Need python$PY_VERSION (or install uv: https://docs.astral.sh/uv/)" >&2
    exit 1
  fi
  "$PYTHON" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
  PIP=("$VENV/bin/python" -m pip install)
fi

"${PIP[@]}" "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
"${PIP[@]}" -r "$HERE/requirements.txt"

"$VENV/bin/python" - <<'EOF'
import mujoco, torch, rsl_rl, tensordict
print(f"mujoco {mujoco.__version__}, torch {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
EOF
echo "Environment ready: $VENV/bin/python"
