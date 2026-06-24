# gemma-vllm-speedtest

RunPod 같은 새 GPU 박스에서 **`git clone` → `bash run.sh`** 한 방으로
Gemma **26B / 31B** (vLLM) 의 **풀-컨텍스트 속도**를 측정하는 셀프-컨테인드 레포.

설치(vLLM 0.20.2, 프로덕션과 동일 핀) → 모델별 서버 기동 → health 대기 →
거의 최대 컨텍스트로 prefill+decode 측정 → 서버 종료 → `results/*.json` 저장까지 자동.

---

## 빠른 시작

```bash
git clone <YOUR_REPO_URL> gemma-vllm-speedtest
cd gemma-vllm-speedtest

# 31B 는 게이트 모델 → 토큰 필요 (26B 만 돌리면 불필요)
export HF_TOKEN=hf_xxx          # huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct 라이선스 동의 먼저

bash run.sh both                # 26b → 31b 순서로 측정 (기본값)
# bash run.sh 26b               # 26B 만
# bash run.sh 31b               # 31B 만
```

끝나면 콘솔에 요약 표가 뜨고, 상세는 `results/<model>_<ts>.json`, 서버 로그는 `results/server_*.log`.

---

## 요구사항

- **GPU 1장** (이 모델들 + 32k 컨텍스트엔 80GB급 A100/H100 권장. 24~48GB면 `MAX_MODEL_LEN`·`GPU_UTIL` 낮춰야 함)
- **디스크** 모델당 ~15–22GB (HF 캐시는 `/workspace` 있으면 거기, 없으면 `./.cache/huggingface`)
- python3 ≥ 3.10 (설치 시 `uv` 로 3.11 venv 자동 생성, 없으면 시스템 python 으로 폴백)
- 인터넷(모델/휠 다운로드). `curl` 은 있으면 `uv` 설치에 쓰이지만 **필수 아님**(health 체크는 venv 파이썬 사용)
- 31B: **`HF_TOKEN`** + HF 라이선스 동의

---

## 무엇을 재는가

요청 프롬프트를 **토큰 ID 리스트로 직접 전송**(재토큰화 드리프트 없음) + 요청마다 랜덤 토큰
(prefix-cache 단축 방지) + 스트리밍으로 측정:

| 지표 | 의미 |
|---|---|
| `ttft_s` | Time-To-First-Token — 긴 프롬프트 **prefill** 속도 반영 |
| `tpot_ms` | 토큰간 지연(ITL) — **decode** 1토큰당 시간 |
| `dec_tok/s/req` | 요청 1개의 decode 처리량 |
| `out_tok/s(agg)` | 동시 요청 전체의 출력 토큰 처리량(집계) |
| `tot_tok/s(agg)` | (입력+출력) 집계 처리량 |
| `req/s` | 초당 완료 요청 수 |

`ignore_eos=true` 로 출력 길이를 정확히 `OUTPUT_LEN` 으로 고정해 측정 일관성 확보.

> **측정 해석 주의**
> - `dec_tok/s/req`, `tpot_ms` 는 **첫토큰~마지막토큰 구간** 기준 → 순수 decode 속도(가장 신뢰).
> - `out_tok/s(agg)`, `tot_tok/s(agg)` 는 요청 묶음 전체의 **end-to-end(=prefill+decode 포함)** 집계라
>   ramp-up/down 이 섞여 steady-state 보다 보수적으로 나옴. 동시성 비교용 지표.
> - `ENFORCE_EAGER=1`(기본) 은 **프로덕션과 동일**(이 모델들은 프로덕션에서 `--enforce-eager` 로 서빙).
>   CUDA graph 를 켠 best-case decode 가 궁금하면 `ENFORCE_EAGER=0` 으로 비교.

---

## 노브 (환경변수로 override)

| 변수 | 기본 | 설명 |
|---|---|---|
| `MAX_MODEL_LEN` | `32768` | 컨텍스트 길이. 입력은 자동으로 `MAX_MODEL_LEN-OUTPUT_LEN-128` 만큼 채움(거의 풀) |
| `INPUT_LEN` | 자동 | 직접 지정 시 우선 |
| `OUTPUT_LEN` | `256` | 생성 토큰 수(고정) |
| `CONCURRENCY` | `1,2,4` | 동시성 레벨(콤마) |
| `GPU_UTIL` | `0.90` | `--gpu-memory-utilization` |
| `MAX_NUM_SEQS` | `16` | 동시 시퀀스 상한 |
| `MAX_NUM_BATCHED_TOKENS` | `=MAX_MODEL_LEN` | prefill 배치 토큰 예산(기본=컨텍스트 → full prompt 가 한 배치에) |
| `ENFORCE_EAGER` | `1` | **1=프로덕션과 동일(CUDA graph off)**. `0` 으로 주면 보통 decode 더 빠름 → 비교용 |
| `GPU` | `0` | 사용할 GPU 인덱스 |
| `PORT` | `8000` | vLLM 포트(127.0.0.1) |
| `HF_TOKEN` | — | 게이트(31B) 모델용 |
| `HF_HOME` | `./.cache/huggingface` | 모델 캐시 위치(영속 볼륨으로 바꾸면 재다운로드 회피) |
| `SKIP_INSTALL` | `0` | 설치 단계 건너뜀 |
| `REINSTALL` | `0` | 강제 재설치 |
| `HEALTH_TIMEOUT` | `1800` | 서버 기동 대기 한도(초) |

예) 64k 컨텍스트로 단일 스트림만:
```bash
MAX_MODEL_LEN=65536 CONCURRENCY=1 bash run.sh 31b
```
예) CUDA graph 켠 best-case decode 비교:
```bash
ENFORCE_EAGER=0 bash run.sh 26b
```

---

## 구조

```
run.sh                # 엔트리포인트: 설치→서버→health→벤치→종료
scripts/install.sh    # vLLM 0.20.2 환경(.venv) — 프로덕션과 동일 핀
bench/benchmark.py    # async 벤치 클라이언트(스트리밍, 토큰ID 프롬프트), 결과에 GPU 메타 기록
bench/compare.py      # 여러 GPU 결과 JSON → CLI 비교 표 + 가성비 리더보드
gpu-bench/build.py    # 결과 JSON → 예쁜 웹 페이지(index.html) 생성
gpu-bench/index.html  # 생성된 웹 리포트(GitHub Pages 로 공개 가능)
configs/26b.env       # 26B 모델 정의
configs/31b.env       # 31B 모델 정의(게이트)
results/              # 산출물(JSON/서버로그) — git 미추적
```

## GPU 비교 / 웹 리포트

여러 GPU에서 같은 벤치를 돌려 비교하려면, 각 박스에서 **가격까지 기록**해서 돌리고:
```bash
GPU_PRICE_USD_HR=0.836 ./run.sh both     # GPU 이름·VRAM·드라이버·가격이 결과 JSON에 박힘
```
- **CLI 비교**: `python bench/compare.py results/*.json` → GPU×동시성 표 + decode/가성비 리더보드
- **웹 리포트**: 결과 JSON을 `gpu-bench/data/` 에 모으고 `python gpu-bench/build.py` → `gpu-bench/index.html`
  (생성 결과·데이터(`index.html`/`data/*.json`)는 `.gitignore` — **로컬 전용 뷰**, 푸시 안 됨)

자세한 흐름은 [gpu-bench/README.md](gpu-bench/README.md).

---

## 트러블슈팅 (RunPod에서 Claude로 빠르게)

- **31B 다운로드 401/403** → `HF_TOKEN` 누락 또는 라이선스 미동의.
- **`fp8e4nv not supported in this architecture`** → fp8 KV 시도 시 A100(Ampere)에선 불가(H100 전용). 이 레포 기본값은 fp8 미사용.
- **OOM / KV 부족(`estimated maximum model length ...`)** → `GPU_UTIL` 또는 `MAX_MODEL_LEN` 낮추기.
- **health 타임아웃** → `results/server_*.log` 확인. 첫 실행은 다운로드로 오래 걸림(`HEALTH_TIMEOUT` 상향).
- **포트 충돌** → `PORT` 변경.

> 이 모델들은 `sliding_window=1024` 혼합 어텐션이라 긴 컨텍스트에서 토큰당 KV가
> 선형 추정보다 낮게 듭니다. 큰 GPU면 `MAX_MODEL_LEN` 을 더 키워볼 만합니다.
