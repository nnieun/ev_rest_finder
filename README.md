# 전기 화물차 구간 전력 소비량 예측 · 휴게소 충전소 추천

Volvo FH Electric 공식 제원(사용가능 460kWh · 최대주행 470km · CCS 350kW · 최대총중량 65t)을
기준으로, 구간 전력 소비량(kWh/100km)을 회귀 모델로 예측하고 그 결과로 고속도로 휴게소
충전소를 추천하는 프로젝트입니다.

노트북에서 분석·모델을 만들고, 웹앱(Streamlit)은 그 결과물만 읽어서 서비스합니다.

## 실행 방법

```bash
uv sync
uv run streamlit run streamlit_app.py
```

`.env`에 아래 키가 필요합니다 (`.env` 파일에 이미 항목만 마련돼 있음):

| 키 | 용도 |
|---|---|
| `KAKAO_REST_API_KEY` | 길찾기·지오코딩·주변 검색 (카카오 로컬/모빌리티 API) |
| `KAKAO_JS_KEY` | (현재 미사용 — 지도는 folium/OpenStreetMap으로 대체) |
| `DATA_GO_KR_KEY` | 전기차 충전소 실시간 상태 (`B552584/EvCharger`) |

## 파일 구성

### 노트북 — 실행 순서대로

1. **`route_segmentation.ipynb`** — 카카오 길찾기 API로 부산 물류센터 → 인천공항 경로를
   받아 휴게소 단위로 구간을 나누고, 구간별 거리·평균속도를 계산.
   → `busan_incheon_route_segments.csv`, `busan_incheon_rest_areas_ev.csv` 생성.
2. **`charger_status_prep.ipynb`** — data.go.kr 전기차 충전소 정보 API에서
   `kind=C0`(고속도로/자동차전용도로 관련 시설) 필터로 전국 고속도로 휴게소·영업소
   670곳만 추려 위경도 인덱스로 저장. 카카오 이름 검색 방식보다 정확하고, 같은 API로
   실시간 혼잡도까지 붙일 수 있게 한다.
   → `data/highway_rest_area_chargers.csv` 생성.
3. **`ev_energy_ml_report.ipynb`** — 이 프로젝트의 핵심 분석 노트북.
   - 02장: `data/ev_energy_consumption.csv`(8,000건 합성 주행데이터) 탐색, 파생변수
     5종(`payload_grade` 등) 설계, 기온·배터리온도 심층분석
   - 03장: 9개 회귀 모델(Linear/Ridge/Lasso/Decision Tree/Random Forest/XGBoost +
     튜닝 3종) 비교 → **Lasso 채택** (Test R²=0.944, MAE=0.68kWh, RMSE=0.85kWh)
   - 04장: 물리 모델(질량·속도·경사) × ML 보정계수(기온·운전성향) 결합, 시나리오 시뮬레이션
   - 06장: 웹앱이 쓸 최종 산출물 저장 → `models/energy_lasso_pipeline.joblib`,
     `models/ml_results.json`
4. **`soc_simulation.ipynb`**, **`rest_stop_planning.ipynb`** — 물리 모델 기반 SOC
   시뮬레이션과 법정 휴게시간(2시간 연속운전당 15분)을 반영한 정차 계획 검토. 웹앱의
   "출발→도착 경로" 기능이 이 로직을 다시 구현해서 쓴다.

### 웹앱

- **`streamlit_app.py`** — 유일한 파이썬 소스 파일. 노트북들이 만든 산출물
  (`models/*.joblib`, `models/*.json`, `data/highway_rest_area_chargers.csv`,
  `busan_incheon_*.csv`)만 읽어서 동작하며, 자체적으로 모델을 재학습하지 않는다.
  - **📍 현재 위치 기준 추천**: 주소 입력 → 근처 휴게소를 예측 소모전력 순으로 추천,
    data.go.kr 실시간 혼잡도 표시. "현재 고속도로입니다" 체크 시 고속도로 시설로만 제한,
    끄면 카카오 검색으로 일반 충전소까지 포함.
  - **🗺️ 출발→도착 경로 기준 추천**: 카카오 길찾기로 실제 도로 폴리라인을 받아 경로 위
    휴게소만 대조하고, SOC 그리디 시뮬레이션으로 몇 회·어디서 충전해야 하는지 계산.
  - **📊 더보기**: `ev_energy_ml_report.ipynb`의 모델 비교·특성중요도를 앱 안에서 재확인.

### 데이터

- `data/ev_energy_consumption.csv` — 모델 학습용 합성 주행 데이터(8,000건).
- `data/highway_rest_area_chargers.csv` — `charger_status_prep.ipynb` 산출물(전국 670곳).
- `data/한국전력공사_*.csv` — 참고용 원본 공공데이터(현재 앱 로직에서는 미사용).
- `busan_incheon_route_segments.csv`, `busan_incheon_rest_areas_ev.csv` —
  `route_segmentation.ipynb` 산출물, 예시 경로(부산→인천공항) 데이터.

### 산출물

- `models/energy_lasso_pipeline.joblib` — 최종 회귀 모델(스케일러+Lasso).
- `models/ml_results.json` — 모델 비교표·특성중요도·산점도 샘플(웹앱 "더보기"용).
- `docs/전기화물차_전비예측_ML_보고서.pptx` — 분석 결과 보고서.
- `docs/전기화물차_휴식장소추천_기획서.docx` — 프로젝트 기획서.

## 알려진 한계

- **경사도 미반영.** 경로에 고도(DEM) 데이터가 연동돼 있지 않아 `road_grade_pct=0`(평지)로
  고정한다. 특성중요도 1위 요인(`payload_grade`, 29%)이 구배를 포함하는데 이 부분이 빠져
  있어, 산악 구간은 실제보다 적게 예측된다. 브이월드(VWorld) 고도 API 연동이 다음 과제.
- **학습 데이터가 승용차 스케일 합성 데이터.** 적재량 0~500kg, 전비 평균 24kWh/100km로
  실제 트럭(28t, ~98kWh/100km)과 스케일이 다르다. 그래서 물리 모델이 절대 수준을 잡고
  ML은 상대 배율만 담당하는 방식으로 우회했다(`ev_energy_ml_report.ipynb` 04장 참고).
- **실시간 혼잡도는 고속도로 인덱스(670곳)에 한해서만 연동.** data.go.kr 충전소 API가
  위치 반경 검색을 지원하지 않고 정확한 충전소 ID로만 조회가 가능해서, "현재 고속도로
  아님" 모드로 찾은 임의의 충전소는 좌표가 겹치지 않으면 혼잡도를 붙이지 못한다.
