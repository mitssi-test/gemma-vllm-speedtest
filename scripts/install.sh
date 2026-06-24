#!/usr/bin/env bash
# vLLM 0.20.2 환경 설치 (프로덕션과 동일한 핀). RunPod 등 새 박스에서 1회 실행.
# - .venv 생성(python 3.11) 후 torch(cu128)/vllm/flashinfer 등 설치
# - uv 가 있으면(또는 설치되면) 빠르게 + python 3.11 자동 확보, 없으면 venv+pip 폴백
# - 이미 설치돼 있으면(.venv/.installed) 건너뜀 (REINSTALL=1 강제)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
MARKER="$VENV/.installed"
PY_BIN="$VENV/bin/python"

VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.20.2/vllm-0.20.2%2Bcu129-cp38-abi3-manylinux_2_31_x86_64.whl"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

log(){ printf '\033[1;36m[install]\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m[install]\033[0m %s\n' "$*"; }

if [ -f "$MARKER" ] && [ "${REINSTALL:-0}" != "1" ]; then
  log "이미 설치됨 ($MARKER). 건너뜀. (강제: REINSTALL=1)"
  if "$PY_BIN" -c "import vllm,torch,transformers" 2>/dev/null; then exit 0; fi
  warn "검증 실패 → 재설치"; rm -f "$MARKER"
fi

# ---- uv 확보 (여러 설치 경로 견고하게 탐색) --------------------------------
ensure_uv(){
  local c
  if c="$(command -v uv 2>/dev/null)"; then printf '%s' "$c"; return 0; fi
  python3 -m pip install --quiet --upgrade uv >/dev/null 2>&1 \
    || python3 -m pip install --quiet --user --upgrade uv >/dev/null 2>&1 \
    || python3 -m pip install --quiet --break-system-packages --upgrade uv >/dev/null 2>&1 \
    || true
  for c in "$(command -v uv 2>/dev/null || true)" "$HOME/.local/bin/uv" \
           "$(python3 -c 'import sysconfig,os;print(os.path.join(sysconfig.get_path("scripts"),"uv"))' 2>/dev/null || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then printf '%s' "$c"; return 0; fi
  done
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | sh >/dev/null 2>&1 || true
    for c in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
      if [ -x "$c" ]; then printf '%s' "$c"; return 0; fi
    done
  fi
  printf ''
}

UV="$(ensure_uv)"
declare -a PIP
if [ -n "$UV" ] && "$UV" venv --python 3.11 "$VENV" 2>/dev/null; then
  log "uv venv 생성 (python 3.11) — $UV"
  # unsafe-best-match: flashinfer 등은 PyPI, torch 는 cu128 인덱스 — uv가 여러 인덱스에서
  # 최적 버전을 찾도록(기본 first-index 면 cu128 에서 flashinfer 버전 못 찾고 실패).
  PIP=( "$UV" pip install --python "$PY_BIN" --index-strategy unsafe-best-match )
else
  [ -n "$UV" ] && warn "uv venv 실패 → python -m venv 폴백" || warn "uv 없음 → python -m venv 폴백"
  PYEXE="$(command -v python3.11 || true)"; [ -n "$PYEXE" ] || PYEXE="$(command -v python3)"
  PYV="$("$PYEXE" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
  case "$PYV" in
    3.11) : ;;
    3.10|3.12|3.13) warn "python $PYV 사용 — 프로덕션 핀은 3.11. abi3 wheel 이라 동작은 하나 완전 동일 환경은 아님." ;;
    *) echo "ERROR: python $PYV — 3.10~3.13 필요(vllm 0.20.2 abi3 wheel)." >&2; exit 1 ;;
  esac
  log "venv 생성 ($PYEXE, python $PYV)"
  "$PYEXE" -m venv "$VENV"
  "$PY_BIN" -m pip install --quiet --upgrade pip
  PIP=( "$PY_BIN" -m pip install )
fi

# ---- 패키지 설치 (프로덕션 install_vllm.sh 와 동일 버전) -------------------
log "[1/3] torch/vision/audio 2.11.0 (cu128 전용 인덱스)"
"${PIP[@]}" torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url "$TORCH_INDEX"

# 중요: vllm wheel 설치 시 torch 가 PyPI 의 일반 빌드로 교체되지 않도록 cu128 인덱스를
#       extra 로 열어두고 torch 핀을 재명시(이미 충족이면 그대로 유지됨).
log "[2/3] vllm 0.20.2 + transformers/flashinfer/compressed-tensors (cu128 torch 보호)"
"${PIP[@]}" "$VLLM_WHEEL" \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  transformers==5.9.0 flashinfer-python==0.6.8.post1 compressed-tensors==0.15.0.1 \
  --extra-index-url "$TORCH_INDEX"

log "[3/3] fastapi/starlette/uvicorn 핀 (vllm 0.20.2 호환)"
"${PIP[@]}" 'fastapi==0.136.3' 'starlette==1.2.1' 'uvicorn==0.48.0'

log "검증 (torch cu128 빌드 확인 포함)"
"$PY_BIN" - <<'PY'
import torch, vllm, transformers, aiohttp
cu = torch.version.cuda
assert torch.__version__.startswith("2.11.0"), f"torch {torch.__version__} (expected 2.11.0)"
assert cu and cu.startswith("12.8"), f"torch CUDA build {cu} (expected 12.8 / cu128)"
print(f"OK vllm {vllm.__version__} | torch {torch.__version__} (cuda {cu}) | transformers {transformers.__version__}")
PY

touch "$MARKER"
log "완료 → $VENV"
