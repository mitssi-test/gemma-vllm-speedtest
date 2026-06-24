# Gemma vLLM 속도 테스트 — 사전설치 이미지
# deps(vLLM 0.20.2 + torch cu128 + flashinfer 등)를 .venv 에 구워둠 → vast.ai 등에서 설치 0초.
# 모델은 라이선스/용량 때문에 굽지 않음(런타임 다운로드, 게이트 31B 는 HF_TOKEN 필요).
# 빌드는 .github/workflows/docker-image.yml 가 GitHub Actions 에서 ghcr.io 로 push.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/gemma-vllm-speedtest
COPY . .

# 프로덕션 핀 그대로 설치 (uv 가 python 3.11 확보). GPU 없이 빌드돼도 OK(휠 다운로드만).
RUN bash scripts/install.sh

# 런타임: 설치 건너뜀(이미 .venv 구워짐). 모델 캐시는 런타임 마운트 권장.
ENV SKIP_INSTALL=1 \
    HF_HOME=/workspace/.hf_home

# vast.ai 는 보통 자체 entrypoint(SSH) 로 덮어쓰니, SSH 후 수동 실행:
#   cd /opt/gemma-vllm-speedtest && HF_TOKEN=hf_xxx ./run.sh both
CMD ["bash"]
