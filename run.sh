#!/usr/bin/env bash
# ============================================================================
#  Gemma vLLM 속도 테스트 — 원클릭 엔트리포인트
#  사용:  bash run.sh [26b|31b|both]      (기본: both)
#
#  하는 일: 설치(.venv) → 모델별로 [vLLM 서버 기동 → health 대기 →
#           full-context 벤치 → 서버 종료 → GPU 회수] → results/ 에 JSON 저장.
#
#  주요 환경변수 노브(override 가능):
#    MAX_MODEL_LEN(32768) GPU_UTIL(0.90) MAX_NUM_SEQS(16)
#    MAX_NUM_BATCHED_TOKENS(기본=MAX_MODEL_LEN)  ENFORCE_EAGER(1)
#    OUTPUT_LEN(256)  CONCURRENCY(1,2,4)  PORT(8000)  GPU(0)  API_KEY(speedtest)
#    INPUT_LEN(자동=MAX_MODEL_LEN-OUTPUT_LEN-128)
#    HF_TOKEN(게이트 모델용)  HF_HOME(기본 /workspace 있으면 거기, 없으면 ./.cache)
#    SKIP_INSTALL(0)  REINSTALL(0)  HEALTH_TIMEOUT(1800)
# ============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TARGET="${1:-both}"
VENV="$ROOT/.venv"
PYBIN="$VENV/bin/python"

# ---- 노브 기본값 ----------------------------------------------------------
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-speedtest}"
GPU="${GPU:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
# prefill 한 배치에 full-context 가 들어가도록 기본을 max-model-len 과 동일하게.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
CONCURRENCY="${CONCURRENCY:-1,2,4}"
INPUT_LEN="${INPUT_LEN:-$(( MAX_MODEL_LEN - OUTPUT_LEN - 128 ))}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
# HF 캐시: RunPod 영속 마운트(/workspace) 있으면 거기로(파드 재생성시 재다운로드 회피).
if [ -z "${HF_HOME:-}" ]; then
  if [ -d /workspace ] && [ -w /workspace ]; then HF_HOME=/workspace/.cache/huggingface
  else HF_HOME="$ROOT/.cache/huggingface"; fi
fi
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

log(){ printf '\033[1;32m[run]\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m[run]\033[0m %s\n' "$*"; }
die(){ printf '\033[1;31m[run] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 입력 검증 ------------------------------------------------------------
[ "$INPUT_LEN" -gt 0 ] 2>/dev/null || die "INPUT_LEN=$INPUT_LEN <= 0 (MAX_MODEL_LEN=$MAX_MODEL_LEN 가 OUTPUT_LEN+128 보다 작음). MAX_MODEL_LEN 키우거나 INPUT_LEN 직접 지정."
case "$TARGET" in 26b|31b|both) ;; *) die "알 수 없는 대상 '$TARGET' (26b|31b|both)";; esac

SERVER_PID=""
cleanup(){
  [ -n "$SERVER_PID" ] || return 0
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    # 실제 PGID 를 조회. 스크립트 자신의 그룹과 같으면(예외적 fork) 그룹 kill 금지(자해 방지).
    local pgid mypgid
    pgid="$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')"
    mypgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
    warn "vLLM 서버 종료 (pid $SERVER_PID, pgid ${pgid:-?})"
    if [ -n "$pgid" ] && [ "$pgid" != "$mypgid" ]; then
      kill -INT "-$pgid" 2>/dev/null || true
      for _ in $(seq 1 40); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
      kill -KILL "-$pgid" 2>/dev/null || true
    else
      kill -INT "$SERVER_PID" 2>/dev/null || true
      for _ in $(seq 1 40); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
      kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
  fi
  SERVER_PID=""
}
on_sig(){ warn "신호 수신 → 정리 후 종료"; cleanup; exit 130; }
trap cleanup EXIT
trap on_sig INT TERM

# GPU 메모리가 충분히 회수될 때까지 대기 (모델 2개 연속 실행 시 OOM 방지)
wait_gpu_free(){
  local gpu="$1"
  command -v nvidia-smi >/dev/null 2>&1 || { sleep 10; return; }
  for _ in $(seq 1 60); do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -1 | tr -d ' ')"
    [ -n "$used" ] || { sleep 1; continue; }
    [ "$used" -lt 2000 ] 2>/dev/null && return
    sleep 1
  done
}

# health 체크: curl 의존 제거 — venv 파이썬(urllib)으로 확인
health_ok(){ "$PYBIN" -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=3)" >/dev/null 2>&1; }

# ---- 1) 설치 --------------------------------------------------------------
if [ "${SKIP_INSTALL:-0}" = "1" ]; then
  log "SKIP_INSTALL=1 → 설치 건너뜀"
else
  log "설치/검증 (scripts/install.sh)"
  bash "$ROOT/scripts/install.sh" || die "설치 실패"
fi
[ -x "$PYBIN" ] || die ".venv 파이썬 없음: $PYBIN (SKIP_INSTALL 해제하고 먼저 설치 필요)"

# GPU 용량 경고 (기본값은 ~80GB 가정)
if command -v nvidia-smi >/dev/null 2>&1; then
  TOTMB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | head -1 | tr -d ' ')"
  if [ -n "$TOTMB" ] && [ "$TOTMB" -lt 49000 ] 2>/dev/null; then
    warn "GPU $GPU 총 ${TOTMB}MB (<49GB): 기본 MAX_MODEL_LEN($MAX_MODEL_LEN)/GPU_UTIL($GPU_UTIL) 로 KV 부족/OOM 가능. 낮춰서 실행 권장."
  fi
fi

declare -a RESULT_FILES=()

# ---- helper: 한 모델 실행 --------------------------------------------------
run_one(){
  local cfg="$1"
  [ -f "$cfg" ] || die "config 없음: $cfg"
  local MODEL="" SERVED_NAME=""
  local -a MODEL_EXTRA_ARGS=()
  # shellcheck disable=SC1090
  source "$cfg"
  [ -n "$MODEL" ] || die "$cfg 에 MODEL 미정의"
  [ -n "$SERVED_NAME" ] || die "$cfg 에 SERVED_NAME 미정의"

  local ts; ts="$(date +%Y%m%d_%H%M%S)"
  local slog="$ROOT/results/server_${SERVED_NAME}_${ts}.log"
  local rjson="$ROOT/results/${SERVED_NAME}_${ts}.json"

  log "================= $SERVED_NAME ($MODEL) ================="
  if [[ "$MODEL" == google/* ]] && [ -z "${HF_TOKEN:-}" ]; then
    warn "게이트 모델인데 HF_TOKEN 없음 → 다운로드 401/403 가능 (라이선스 동의+토큰 필요)"
  fi
  if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  fi

  local -a eager=()
  [ "$ENFORCE_EAGER" = "1" ] && eager=(--enforce-eager)

  log "vLLM 서버 기동 (GPU $GPU, port $PORT, max-model-len $MAX_MODEL_LEN, util $GPU_UTIL, batched $MAX_NUM_BATCHED_TOKENS, eager=$ENFORCE_EAGER)"
  log "  서버 로그: $slog"
  # setsid → 자식(EngineCore 등)까지 한 프로세스 그룹 → cleanup 에서 그룹 전체 정리
  CUDA_VISIBLE_DEVICES="$GPU" setsid "$PYBIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --dtype auto --trust-remote-code \
    "${eager[@]}" "${MODEL_EXTRA_ARGS[@]}" \
    --api-key "$API_KEY" \
    >"$slog" 2>&1 &
  SERVER_PID=$!

  # health 대기 (벽시계 기준)
  log "health 대기 (최대 ${HEALTH_TIMEOUT}s — 첫 실행은 모델 다운로드로 오래 걸림)"
  local deadline=$(( $(date +%s) + HEALTH_TIMEOUT )) ready=0
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      warn "서버 프로세스가 죽음 — 로그 마지막:"; tail -n 30 "$slog" >&2; cleanup; wait_gpu_free "$GPU"; return 1
    fi
    if health_ok; then ready=1; break; fi
    sleep 2
  done
  [ "$ready" = "1" ] || { warn "health 타임아웃 — 로그 마지막:"; tail -n 30 "$slog" >&2; cleanup; wait_gpu_free "$GPU"; return 1; }
  log "서버 준비 완료 → 벤치 시작"

  # 벤치
  "$PYBIN" "$ROOT/bench/benchmark.py" \
    --base-url "http://127.0.0.1:$PORT" --api-key "$API_KEY" \
    --model "$SERVED_NAME" \
    --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" \
    --concurrency "$CONCURRENCY" \
    --label "$SERVED_NAME (len=$MAX_MODEL_LEN, eager=$ENFORCE_EAGER)" \
    --output "$rjson"
  local rc=$?

  RESULT_FILES+=("$rjson")
  cleanup
  wait_gpu_free "$GPU"   # 다음 모델 전에 VRAM 회수 확인
  log "$SERVED_NAME 완료 (rc=$rc). 결과: $rjson"
  return $rc
}

# ---- 2) 대상 선택 후 실행 --------------------------------------------------
declare -a CFGS
case "$TARGET" in
  26b)  CFGS=("$ROOT/configs/26b.env") ;;
  31b)  CFGS=("$ROOT/configs/31b.env") ;;
  both) CFGS=("$ROOT/configs/26b.env" "$ROOT/configs/31b.env") ;;
  *) die "알 수 없는 대상 '$TARGET' (26b|31b|both)" ;;
esac

log "대상: $TARGET | input_len=$INPUT_LEN output_len=$OUTPUT_LEN concurrency=$CONCURRENCY | HF_HOME=$HF_HOME"
FAIL=0
for c in "${CFGS[@]}"; do
  run_one "$c" || { warn "$(basename "$c") 실패"; FAIL=1; }
done

# ---- 3) 이번 실행 결과만 요약 ---------------------------------------------
echo
log "================= 이번 실행 결과 요약 ================="
for f in "${RESULT_FILES[@]:-}"; do
  [ -n "$f" ] && [ -e "$f" ] || continue
  "$PYBIN" - "$f" <<'PY' 2>/dev/null || true
import json,sys
d=json.load(open(sys.argv[1]))
print(f"\n# {d['label']}  (model={d['model']})")
print(f"  input_len={d['input_len']} output_len={d['output_len']}")
for r in d["results"]:
    print(f"   conc={r['concurrency']:>2}  ttft={r['ttft_s_mean']}s  tpot={r['tpot_ms_mean']}ms  "
          f"out_tok/s(agg)={r['output_tok_s_agg']}  dec_tok/s/req={r['decode_tok_s_per_req_mean']}  "
          f"req/s={r['req_per_s']}  ok={r['ok']}/{r['requests']}")
PY
done
echo
[ "$FAIL" = "0" ] && log "DONE ✅" || { warn "일부 실패 — 위 로그/results 확인"; exit 1; }
