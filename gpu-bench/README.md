# gpu-bench — 결과를 웹으로 예쁘게 (로컬 전용 뷰)

벤치 결과 JSON들을 모아 **단일 HTML 페이지**(`index.html`)로 렌더링합니다.
GPU × 모델(26B/31B) × 모드(eager / cudagraph)별 표 + 라인차트(동시성축) + 가성비(tok/$).

## 푸시 정책 (중요)

- **공유(git push)되는 것**: 도구 `build.py` 와 이 README 뿐.
- **로컬 전용(.gitignore, 푸시 안 됨)**: 생성된 `index.html` 과 `data/*.json`.
  → 결과/페이지는 **이 머신에서만** 보고, GitHub엔 안 올라갑니다.

다른 머신에서도 `build.py` 만 받으면 거기 자기 결과로 페이지를 만들 수 있습니다(데이터는 각자 로컬).

## 생성

```bash
# results/ (로컬 실행분) + gpu-bench/data/ (모아둔 분) 자동으로 읽음
python gpu-bench/build.py

# 경로 직접 지정도 가능
python gpu-bench/build.py results/*.json gpu-bench/data/*.json --title "내 벤치"
```
→ `gpu-bench/index.html` 갱신. 브라우저로 그 파일 열면 끝(차트는 Chart.js CDN).

## 여러 GPU 비교 (로컬)

각 박스에서 가격까지 기록해 돌리고:
```bash
GPU_PRICE_USD_HR=0.836 ./run.sh both      # GPU 이름·VRAM·드라이버·가격이 결과 JSON에 박힘
```
박스가 사라지기 전에 그 `results/*.json` 을 **이 머신의 `gpu-bench/data/`** 로 가져오면(scp/다운로드 등),
한 페이지에 GPU들이 나란히 비교됩니다:
```bash
python gpu-bench/build.py        # gpu-bench/data/ 의 모든 GPU가 한 페이지에
```
- 같은 GPU의 eager(`./run.sh`)와 cudagraph(`ENFORCE_EAGER=0 ./run.sh`) 둘 다 넣으면 한 차트에 두 선.
- `data/*.json` 은 git에 안 올라가니, 백업하려면 따로 보관하세요.

> 나중에 웹 공개를 원하면 `.gitignore` 에서 `gpu-bench/index.html`·`gpu-bench/data/*.json` 줄을 빼고
> 커밋·푸시한 뒤 GitHub Pages(Settings→Pages→main/root)로 `…/gpu-bench/` 서빙하면 됩니다.
