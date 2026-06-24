#!/usr/bin/env python3
"""vLLM OpenAI 호환 서버 속도 벤치마크 (full-context 위주).

특징:
- 프롬프트를 '토큰 ID 리스트'로 직접 전송 → 재토큰화 드리프트 없이 정확한 입력 길이.
  (transformers/HF 접근 불필요 → 게이트 모델이어도 클라이언트는 토크나이저가 필요 없음)
- 요청마다 다른 랜덤 토큰 → prefix-cache 단축 방지(공정한 prefill 측정).
- 스트리밍으로 TTFT / ITL / decode tok/s 측정, ignore_eos 로 출력 길이 고정.
- concurrency 레벨별로 반복 → 동시성 확장 측정.

측정 지표(레벨별):
  TTFT(첫 토큰까지), TPOT/ITL(토큰간 지연), per-req decode tok/s,
  집계 output tok/s, 집계 total tok/s, req/s.
"""
import argparse
import asyncio
import json
import math
import os
import random
import statistics
import sys
import time

import aiohttp


def gather_env(price_usd_hr=None):
    """벤치 박스 환경 메타(특히 GPU) 수집 — GPU 간 비교에 필수."""
    env = {}
    try:
        from importlib.metadata import version, PackageNotFoundError
        for pkg in ("vllm", "torch", "transformers", "flashinfer-python"):
            try:
                env[pkg] = version(pkg)
            except PackageNotFoundError:
                pass
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        gpus = [ln.strip() for ln in out if ln.strip()]
        if gpus:
            parts = [p.strip() for p in gpus[0].split(",")]
            env["gpu_name"] = parts[0] if parts else None
            if len(parts) > 1:
                env["gpu_mem"] = parts[1]
            if len(parts) > 2:
                env["driver"] = parts[2]
            env["gpu_count"] = len(gpus)
    except Exception:
        pass
    env["python"] = "%d.%d.%d" % sys.version_info[:3]
    if price_usd_hr:
        try:
            env["price_usd_hr"] = float(price_usd_hr)
        except (TypeError, ValueError):
            pass
    return env


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


async def one_request(session, url, headers, model, prompt_ids, output_len, idx, results):
    payload = {
        "model": model,
        "prompt": prompt_ids,           # list[int] — 정확한 입력 토큰 수
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,             # vLLM 확장: EOS 무시하고 정확히 output_len 생성
        "add_special_tokens": False,    # BOS 미추가 → prompt_tokens == input_len 정확히
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    tok_times = []
    prompt_tokens = None
    completion_tokens = None
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                results.append({"idx": idx, "error": f"HTTP {resp.status}: {body[:300]}"})
                return
            async for raw in resp.content:
                line = raw.strip()
                if not line or not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                choices = obj.get("choices") or []
                if choices and choices[0].get("text"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    tok_times.append(now)
                usage = obj.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
    except Exception as e:
        results.append({"idx": idx, "error": repr(e)})
        return

    t_end = time.perf_counter()
    e2e = t_end - t0
    comp = completion_tokens if completion_tokens else len(tok_times)
    # ITL: 첫 토큰 이후 토큰간 간격
    itls = [tok_times[i] - tok_times[i - 1] for i in range(1, len(tok_times))]
    # decode tok/s: 첫토큰~마지막토큰 구간(comp-1 간격) 기준 → 트레일링 usage/DONE 왕복 제외.
    if len(tok_times) >= 2:
        span = tok_times[-1] - tok_times[0]
        decode_tok_s = (len(tok_times) - 1) / span if span > 0 else 0.0
    else:
        decode_tok_s = 0.0
    results.append({
        "idx": idx,
        "ttft": ttft,
        "e2e": e2e,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": comp,
        "decode_tok_s": decode_tok_s,
        "tpot_ms": (statistics.mean(itls) * 1000) if itls else None,
    })


async def run_level(base_url, headers, model, input_len, output_len,
                    concurrency, num_prompts, vocab_lo, vocab_hi, seed_base):
    url = base_url.rstrip("/") + "/v1/completions"
    sem = asyncio.Semaphore(concurrency)
    results = []
    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=3600)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def worker(i):
            async with sem:
                rng = random.Random(seed_base + i)
                ids = [rng.randint(vocab_lo, vocab_hi) for _ in range(input_len)]
                await one_request(session, url, headers, model, ids, output_len, i, results)

        t0 = time.perf_counter()
        await asyncio.gather(*[worker(i) for i in range(num_prompts)])
        wall = time.perf_counter() - t0
    return results, wall


def summarize(results, wall, concurrency, input_len, output_len):
    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    decs = [r["decode_tok_s"] for r in ok]
    tpots = [r["tpot_ms"] for r in ok if r.get("tpot_ms") is not None]
    tot_out = sum(r["completion_tokens"] or 0 for r in ok)
    tot_in = sum((r["prompt_tokens"] or input_len) for r in ok)
    return {
        "concurrency": concurrency,
        "input_len": input_len,
        "output_len": output_len,
        "requests": len(results),
        "ok": len(ok),
        "errors": len(errs),
        "error_samples": [e["error"] for e in errs[:3]],
        "wall_s": round(wall, 3),
        "ttft_s_mean": round(statistics.mean(ttfts), 3) if ttfts else None,
        "ttft_s_p50": round(pct(ttfts, 50), 3) if ttfts else None,
        "ttft_s_p99": round(pct(ttfts, 99), 3) if ttfts else None,
        "tpot_ms_mean": round(statistics.mean(tpots), 2) if tpots else None,
        "decode_tok_s_per_req_mean": round(statistics.mean(decs), 1) if decs else None,
        "output_tok_s_agg": round(tot_out / wall, 1) if wall > 0 else None,
        "total_tok_s_agg": round((tot_in + tot_out) / wall, 1) if wall > 0 else None,
        "req_per_s": round(len(ok) / wall, 3) if wall > 0 else None,
    }


def print_table(label, rows):
    cols = [
        ("conc", "concurrency", 4),
        ("in", "input_len", 7),
        ("out", "output_len", 5),
        ("ok", "ok", 4),
        ("err", "errors", 4),
        ("ttft_s", "ttft_s_mean", 8),
        ("tpot_ms", "tpot_ms_mean", 8),
        ("dec_tok/s", "decode_tok_s_per_req_mean", 10),
        ("out_tok/s", "output_tok_s_agg", 10),
        ("tot_tok/s", "total_tok_s_agg", 10),
        ("req/s", "req_per_s", 7),
    ]
    print(f"\n=== {label} ===")
    header = " ".join(f"{h:>{w}}" for h, _, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = " ".join(f"{('' if r.get(k) is None else r.get(k)):>{w}}" for _, k, w in cols)
        print(line)
    print()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", "speedtest"))
    ap.add_argument("--model", required=True, help="served-model-name")
    ap.add_argument("--input-len", type=int, required=True)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--concurrency", default="1", help="콤마 구분 레벨, 예: 1,2,4")
    ap.add_argument("--num-prompts-per-level", default="auto",
                    help="'auto' = max(concurrency,3), 또는 정수")
    ap.add_argument("--vocab-lo", type=int, default=1000)
    ap.add_argument("--vocab-hi", type=int, default=50000)
    ap.add_argument("--label", default="")
    ap.add_argument("--output", default="", help="결과 JSON 저장 경로")
    ap.add_argument("--price-usd-hr", default=os.environ.get("GPU_PRICE_USD_HR", ""),
                    help="시간당 GPU 가격(USD) — 비교 시 tok/$ 계산용(선택)")
    args = ap.parse_args()

    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    levels = [int(x) for x in str(args.concurrency).split(",") if x.strip()]
    label = args.label or args.model

    env = gather_env(args.price_usd_hr)
    if env.get("gpu_name"):
        print(f"[bench] GPU: {env.get('gpu_name')} x{env.get('gpu_count', 1)} "
              f"({env.get('gpu_mem', '?')}) | vllm {env.get('vllm', '?')} torch {env.get('torch', '?')}",
              flush=True)

    rows = []
    for c in levels:
        if args.num_prompts_per_level == "auto":
            # concurrency 의 배수로 맞춰 'solo tail'(부분 동시성) 측정 왜곡 방지
            n = c * max(1, math.ceil(3 / c))
        else:
            n = int(args.num_prompts_per_level)
        print(f"[bench] {label}: concurrency={c} num_prompts={n} "
              f"input_len={args.input_len} output_len={args.output_len} ...", flush=True)
        results, wall = await run_level(
            args.base_url, headers, args.model, args.input_len, args.output_len,
            c, n, args.vocab_lo, args.vocab_hi, seed_base=c * 100000)
        s = summarize(results, wall, c, args.input_len, args.output_len)
        rows.append(s)
        if s["errors"]:
            print(f"  ! errors={s['errors']} sample={s['error_samples']}", flush=True)
        print(f"  -> ttft={s['ttft_s_mean']}s tpot={s['tpot_ms_mean']}ms "
              f"out_tok/s(agg)={s['output_tok_s_agg']} dec_tok/s/req={s['decode_tok_s_per_req_mean']}",
              flush=True)

    print_table(label, rows)

    out = {
        "label": label,
        "model": args.model,
        "base_url": args.base_url,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "enforce_eager": os.environ.get("ENFORCE_EAGER", ""),
        "env": env,
        "results": rows,
    }
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[bench] saved -> {args.output}")

    # 에러만 있고 성공 0 이면 비정상 종료 (run.sh 가 감지)
    if all(r["ok"] == 0 for r in rows):
        print("[bench] 모든 요청 실패", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
