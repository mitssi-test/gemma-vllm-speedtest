#!/usr/bin/env python3
"""여러 GPU에서 나온 결과 JSON들을 모아 GPU 간 비교 표를 출력.

사용:
  python bench/compare.py results/*.json
  python bench/compare.py results/*.json --conc 1          # 특정 동시성만
  python bench/compare.py collected/*.json --csv out.csv    # CSV 로도 저장

각 JSON 은 run.sh/benchmark.py 산출물(env.gpu_name, 선택적 price_usd_hr 포함).
모델별로 GPU×동시성 비교 표 + (가격 있으면) 가성비(1$당 출력토큰) 리더보드를 출력.
"""
import argparse
import glob
import json
import sys


def load(paths):
    recs = []
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"skip {p}: {e}", file=sys.stderr)
            continue
        env = d.get("env", {}) or {}
        gpu = env.get("gpu_name") or "?"
        gc = env.get("gpu_count", 1) or 1
        gpu_label = gpu if gc == 1 else f"{gpu} x{gc}"
        price = env.get("price_usd_hr")
        for r in d.get("results", []):
            out_agg = r.get("output_tok_s_agg")
            tok_per_usd = None
            if price and out_agg:
                try:
                    tok_per_usd = round(out_agg * 3600.0 / float(price))
                except (TypeError, ValueError, ZeroDivisionError):
                    tok_per_usd = None
            recs.append({
                "model": d.get("model"),
                "gpu": gpu_label,
                "price": price,
                "eager": d.get("enforce_eager", ""),
                "input_len": d.get("input_len"),
                "output_len": d.get("output_len"),
                "conc": r.get("concurrency"),
                "ttft": r.get("ttft_s_mean"),
                "tpot": r.get("tpot_ms_mean"),
                "dec": r.get("decode_tok_s_per_req_mean"),
                "out_agg": out_agg,
                "reqs": r.get("req_per_s"),
                "tok_per_usd": tok_per_usd,
                "ok": r.get("ok"),
                "errors": r.get("errors"),
                "src": p,
            })
    return recs


def fmt(v):
    return "" if v is None else str(v)


def print_model_table(model, rows, have_price):
    cols = [("GPU", "gpu", 26, "<"), ("conc", "conc", 4, ">"),
            ("ttft_s", "ttft", 8, ">"), ("tpot_ms", "tpot", 8, ">"),
            ("dec_tok/s", "dec", 10, ">"), ("out_tok/s(agg)", "out_agg", 14, ">"),
            ("req/s", "reqs", 7, ">"), ("ok", "ok", 4, ">"), ("err", "errors", 4, ">")]
    if have_price:
        cols.append(("tok/$", "tok_per_usd", 10, ">"))
    il = rows[0].get("input_len")
    ol = rows[0].get("output_len")
    eager = rows[0].get("eager")
    print(f"\n### {model}   (input_len={il} output_len={ol} eager={eager})")
    header = " ".join(f"{h:{a}{w}}" for h, _, w, a in cols)
    print(header)
    print("-" * len(header))
    rows = sorted(rows, key=lambda r: (str(r["gpu"]), r["conc"] or 0))
    for r in rows:
        print(" ".join(f"{fmt(r.get(k)):{a}{w}}" for _, k, w, a in cols))


def print_leaderboard(model, rows, conc, have_price):
    sel = [r for r in rows if r["conc"] == conc]
    if not sel:
        return
    print(f"\n— {model} @ conc={conc} 리더보드 —")
    by_dec = sorted([r for r in sel if r["dec"] is not None], key=lambda r: -r["dec"])
    print("  decode tok/s/req (빠른 순):")
    for r in by_dec:
        print(f"    {r['dec']:>7}  {r['gpu']}")
    if have_price:
        by_val = sorted([r for r in sel if r["tok_per_usd"] is not None], key=lambda r: -r["tok_per_usd"])
        if by_val:
            print("  가성비 out_tok/$ (높을수록 이득):")
            for r in by_val:
                pr = f"${r['price']}/h" if r['price'] else ""
                print(f"    {r['tok_per_usd']:>10}  {r['gpu']} {pr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="결과 JSON 들 (glob 가능)")
    ap.add_argument("--conc", type=int, default=None, help="이 동시성만 표시")
    ap.add_argument("--csv", default="", help="CSV 저장 경로(선택)")
    args = ap.parse_args()

    paths = []
    for p in args.paths:
        paths.extend(glob.glob(p) if any(c in p for c in "*?[") else [p])
    if not paths:
        print("결과 JSON 없음", file=sys.stderr)
        sys.exit(1)

    recs = load(paths)
    if args.conc is not None:
        recs = [r for r in recs if r["conc"] == args.conc]
    if not recs:
        print("표시할 레코드 없음", file=sys.stderr)
        sys.exit(1)

    have_price = any(r["tok_per_usd"] is not None for r in recs)
    models = sorted({r["model"] for r in recs}, key=lambda m: str(m))
    n_gpu = len({r["gpu"] for r in recs})
    print(f"== GPU 비교: {n_gpu}개 GPU, {len(paths)}개 결과파일, 모델 {len(models)}종 ==")

    for m in models:
        mrows = [r for r in recs if r["model"] == m]
        print_model_table(m, mrows, have_price)
        concs = sorted({r["conc"] for r in mrows if r["conc"] is not None})
        if concs:
            print_leaderboard(m, mrows, args.conc if args.conc is not None else concs[0], have_price)

    if args.csv:
        import csv
        keys = ["model", "gpu", "price", "eager", "input_len", "output_len", "conc",
                "ttft", "tpot", "dec", "out_agg", "reqs", "tok_per_usd", "ok", "errors", "src"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in recs:
                w.writerow(r)
        print(f"\nCSV 저장 -> {args.csv}")


if __name__ == "__main__":
    main()
