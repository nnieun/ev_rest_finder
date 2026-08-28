"""전기 화물차(Volvo FH Electric 기준) 근처 휴게소 충전소 추천 — Streamlit 앱.

모델 학습·검증은 ev_energy_ml_report.ipynb 에서 하고, 이 앱은 그 결과물만 읽는다.
    - models/energy_lasso_pipeline.joblib        최종 회귀 모델 (노트북 06장에서 저장)
    - models/ml_results.json                     모델 비교표 · 특성중요도 · 산점도 샘플 (같은 곳)
    - data/highway_rest_area_chargers.csv         전국 고속도로 휴게소 충전소 위경도
      (charger_status_prep.ipynb 에서 생성 — data.go.kr EvCharger API의 kind=C0 필터로
      "카카오 검색으로 찾고 이름에 '휴게소'가 들어있는지로 판별" 하던 예전 방식을 대체한다.
      진짜 휴게소만 정확히 걸러지고, 같은 API로 충전소 ID 기반 실시간 혼잡도까지 붙는다.)

물리 모델(질량·속도·경사) × ML 보정계수(기온·운전성향) 조합으로 절대 전비를 구하는
이유는 ev_energy_ml_report.ipynb 04장 참고 — 학습 데이터가 승용차 스케일 합성
데이터라 ML 예측값을 그대로 쓸 수 없어서다.
"""
import json
import os
import urllib.parse

import folium
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "energy_lasso_pipeline.joblib")
ML_RESULTS_PATH = os.path.join(BASE_DIR, "models", "ml_results.json")
EV_CSV = os.path.join(BASE_DIR, "data", "ev_energy_consumption.csv")
HIGHWAY_CHARGERS_CSV = os.path.join(BASE_DIR, "data", "highway_rest_area_chargers.csv")

KAKAO_KEY = os.getenv("KAKAO_REST_API_KEY", "")
KAKAO_HEADERS = {"Authorization": f"KakaoAK {KAKAO_KEY}"}

DATA_GO_KR_KEY = urllib.parse.unquote(os.getenv("DATA_GO_KR_KEY", ""))
EVCHARGER_URL = "https://apis.data.go.kr/B552584/EvCharger/getChargerInfo"
STAT_LABELS = {"1": "통신이상", "2": "충전대기(사용가능)", "3": "충전중",
               "4": "운영중지", "5": "점검중", "9": "상태미확인"}

# ── Volvo FH Electric 공개 제원 — 이 앱의 모든 절대 수치는 이 스펙 기준이다 ──
TRUCK_SPEC = {
    "model": "Volvo FH Electric",
    "source": "Volvo Trucks(볼보트럭스) 공식 제원 · 국내 실제 물류 운행 사례 존재",
    "battery_kwh": "360 ~ 540 (배터리팩 4 ~ 6개 구성)",
    "battery_usable_kwh": 460.0,
    "motor_kw": "300 / 350 / 400 / 470 / 540 중 선택",
    "gvw_max_t": 65,          # 볼보 스펙상 최대. 국내 도로법 상한은 40t
    "gvw_kr_limit_t": 40,
    "axle": "4×2 / 6×2 / 6×4",
    "charging": "CCS 최대 350 kW",
    "charge_20_80_min": 65,
    "range_km": 470.0,
}

# ── 물리 상수 (soc_simulation.ipynb / rest_stop_planning.ipynb 와 동일) ──
G, RHO_AIR, CRR, CDA = 9.81, 1.2, 0.006, 5.5
ETA_DRIVE, ETA_REGEN, AUX_KW = 0.85, 0.60, 3.0

CONDITION_SPECS = [
    {"key": "mass_kg", "label": "총중량 (GCW)", "unit": "kg", "min": 18000, "max": 65000,
     "step": 1000, "default": 40000},
    {"key": "ambient_temp_C", "label": "외기 온도", "unit": "°C", "min": -10.0, "max": 40.0,
     "step": 0.5, "default": 15.0},
    {"key": "driving_style_index", "label": "운전 성향 지수", "unit": "0=온화 · 1=공격적",
     "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.35},
]


@st.cache_resource(show_spinner=False)
def load_model():
    artifact = joblib.load(MODEL_PATH)
    ev_df = pd.read_csv(EV_CSV, encoding="utf-8-sig")
    ref = {c: float(ev_df[c].median()) for c in artifact["base_features"]}
    return artifact, ref


ARTIFACT, REF = load_model()
SCALER, ML_MODEL, FEATS = ARTIFACT["scaler"], ARTIFACT["model"], ARTIFACT["features"]


def add_features(row):
    row = dict(row)
    row["speed_sq"] = row["speed_kmh"] ** 2
    row["payload_grade"] = row["payload_kg"] * row["road_grade_pct"]
    row["temp_dev"] = abs(row["ambient_temp_C"] - 21)
    row["hvac_per_speed"] = row["hvac_power_kw"] / row["speed_kmh"]
    row["tire_dev"] = abs(row["tire_pressure_bar"] - 2.5)
    return row


def hvac_from_temp(temp_c):
    return float(np.clip(abs(temp_c - 21) * 0.18, 0, 5))


def _ml_predict(row):
    enriched = add_features(row)
    X = pd.DataFrame([[enriched[f] for f in FEATS]], columns=FEATS)
    X_scaled = pd.DataFrame(SCALER.transform(X), columns=FEATS)  # 학습 때처럼 컬럼명 유지
    return float(ML_MODEL.predict(X_scaled)[0])


def ml_ratio(speed_kmh, temp_c, style):
    row = dict(REF)
    row.update({"speed_kmh": float(speed_kmh), "ambient_temp_C": float(temp_c),
                "driving_style_index": float(style), "hvac_power_kw": hvac_from_temp(temp_c)})
    return _ml_predict(row) / _ml_predict(dict(REF))


def physics_kwh100(v_kmh, mass_kg, grade_pct=0.0):
    v = v_kmh / 3.6
    f = CRR * mass_kg * G + 0.5 * RHO_AIR * CDA * v ** 2 + mass_kg * G * grade_pct / 100
    p = (f * v / ETA_DRIVE if f >= 0 else f * v * ETA_REGEN) + AUX_KW * 1000
    return max(p, 0) / 1000 / v_kmh * 100


def predict_rate(speed_kmh, mass_kg, grade_pct, temp_c, style):
    speed_kmh = max(float(speed_kmh), 5.0)
    return physics_kwh100(speed_kmh, mass_kg, grade_pct) * ml_ratio(speed_kmh, temp_c, style)


# ── 카카오 API (경로/장소 검색용 REST 키) ──────────────────────────────────

def kakao_get(url, params, retries=3):
    if not KAKAO_KEY:
        raise RuntimeError("KAKAO_REST_API_KEY 가 .env 에 없습니다.")
    last = None
    for _ in range(retries):
        r = requests.get(url, headers=KAKAO_HEADERS, params=params, timeout=8)
        if r.status_code == 429:
            last = r
            continue
        r.raise_for_status()
        return r.json()
    last.raise_for_status()


def kakao_geocode(query):
    j = kakao_get("https://dapi.kakao.com/v2/local/search/address.json", {"query": query})
    if j.get("documents"):
        d = j["documents"][0]
        return float(d["x"]), float(d["y"]), d["address_name"]
    j = kakao_get("https://dapi.kakao.com/v2/local/search/keyword.json", {"query": query, "size": 1})
    if not j.get("documents"):
        raise ValueError(f"좌표를 찾을 수 없습니다: {query}")
    d = j["documents"][0]
    return float(d["x"]), float(d["y"]), d["place_name"]


def kakao_directions(origin_xy, dest_xy):
    try:
        j = kakao_get("https://apis-navi.kakaomobility.com/v1/directions",
                      {"origin": f"{origin_xy[0]},{origin_xy[1]}",
                       "destination": f"{dest_xy[0]},{dest_xy[1]}", "priority": "RECOMMEND"})
        route = j["routes"][0]
        if route.get("result_code") != 0:
            return None
        s = route["summary"]
        return s["distance"] / 1000, s["duration"] / 60
    except Exception:
        return None


# ── 전국 고속도로 휴게소 충전소 (data.go.kr, charger_status_prep.ipynb 산출물) ──

@st.cache_resource(show_spinner=False)
def load_highway_chargers():
    return pd.read_csv(HIGHWAY_CHARGERS_CSV, encoding="utf-8-sig")


def haversine_km(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = (np.radians(v) for v in (lng1, lat1, lng2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_highway_chargers(lat, lng, k=8):
    df = load_highway_chargers().copy()
    df["dist_km"] = haversine_km(lng, lat, df["lng"].values, df["lat"].values)
    return df.nsmallest(k, "dist_km").to_dict(orient="records")


def data_go_kr_status(stat_id):
    """statId로 정확히 조회하는 실시간 상태 — 카카오 검색 결과와 이름으로 매칭할 필요가 없다."""
    if not DATA_GO_KR_KEY:
        return {"label": "DATA_GO_KR_KEY 없음", "available": None}
    try:
        r = requests.get(EVCHARGER_URL, params={
            "serviceKey": DATA_GO_KR_KEY, "dataType": "JSON",
            "pageNo": 1, "numOfRows": 20, "statId": stat_id,
        }, timeout=8)
        r.raise_for_status()
        items = r.json().get("items", {}).get("item") or []
        if not items:
            return {"label": "정보 없음", "available": None}
        available = sum(1 for it in items if it.get("stat") == "2")
        if available:
            return {"label": f"충전대기 {available}/{len(items)}", "available": available}
        return {"label": STAT_LABELS.get(items[0].get("stat"), "상태미확인"), "available": 0}
    except Exception:
        return {"label": "조회 실패", "available": None}


@st.cache_data(ttl=600, show_spinner=False)
def cached_geocode(query):
    return kakao_geocode(query)


@st.cache_data(ttl=90, show_spinner=False)
def cached_candidates(lat, lng, k=8):
    """위치당 바뀌지 않는 부분만 캐시: 후보 · 실제 도로거리 · 실시간 혼잡도.
    주행 조건(총중량·기온·운전성향)은 여기 안 넣는다 — 슬라이더를 움직였다고 카카오·
    data.go.kr을 다시 부를 필요는 없다."""
    out = []
    for c in nearest_highway_chargers(lat, lng, k=k):
        result = kakao_directions((lng, lat), (c["lng"], c["lat"]))
        if result is None:
            continue
        distance_km, duration_min = result
        status = data_go_kr_status(c["statId"])
        out.append({
            "name": c["statNm"], "addr": c["addr"], "lat": c["lat"], "lng": c["lng"],
            "driving_km": round(distance_km, 1), "duration_min": round(duration_min, 0),
            "congestion": status["label"], "available": status["available"],
        })
    return out


def rank_with_conditions(candidates, cond):
    ranked = []
    for c in candidates:
        avg_speed = c["driving_km"] / (c["duration_min"] / 60) if c["duration_min"] else 40.0
        rate = predict_rate(avg_speed, cond["mass_kg"], 0.0, cond["ambient_temp_C"],
                             cond["driving_style_index"])
        total_kwh = rate * c["driving_km"] / 100
        ranked.append({**c, "predicted_kwh100km": round(rate, 1),
                       "predicted_total_kwh": round(total_kwh, 2)})
    # 사용 가능한 충전기가 있는 곳을 먼저, 그 안에서는 예측 소모전력이 적은 순.
    ranked.sort(key=lambda r: (0 if (r["available"] or 0) > 0 else 1, r["predicted_total_kwh"]))
    return ranked


# ── 출발지 → 도착지 경로 위의 휴게소 찾기 (route_segmentation.ipynb 와 같은 방식) ──

def kakao_route_polyline(origin_xy, dest_xy):
    """경로의 실제 도로 폴리라인 + 누적거리. route_segmentation_v2 의 route_polyline()과 동일한 방식."""
    try:
        j = kakao_get("https://apis-navi.kakaomobility.com/v1/directions",
                      {"origin": f"{origin_xy[0]},{origin_xy[1]}",
                       "destination": f"{dest_xy[0]},{dest_xy[1]}",
                       "priority": "RECOMMEND", "road_details": True})
        route = j["routes"][0]
        if route.get("result_code") != 0:
            return None
        summary = route["summary"]

        pts = []
        for sec in route["sections"]:
            for road in sec.get("roads", []):
                v = road["vertexes"]
                pts.extend(zip(v[0::2], v[1::2]))
        if not pts:
            return None

        df = pd.DataFrame(pts, columns=["lng", "lat"])
        df = df[(df != df.shift()).any(axis=1)].reset_index(drop=True)
        step = haversine_km(df["lng"].values[:-1], df["lat"].values[:-1],
                            df["lng"].values[1:], df["lat"].values[1:])
        df["cum_km"] = np.concatenate([[0.0], np.cumsum(step)])
        return {"route_df": df, "total_km": float(df["cum_km"].iat[-1]),
                "duration_min": summary["duration"] / 60}
    except Exception:
        return None


def match_rest_areas_to_route(route_df, max_offset_km=2.5):
    """전국 670개 휴게소 인덱스를 경로 폴리라인과 대조해 실제로 경로 위에 있는 곳만 남긴다.
    노트북(route_segmentation)은 매번 카카오에 검색을 던졌지만, 여기선 이미 확보된 로컬
    인덱스와 좌표 거리만 비교하면 되니 API 호출이 필요 없다."""
    stations = load_highway_chargers()
    route_lng, route_lat, route_cum = (route_df[c].values for c in ["lng", "lat", "cum_km"])
    matches = []
    for _, s in stations.iterrows():
        d = haversine_km(s["lng"], s["lat"], route_lng, route_lat)
        k = int(np.argmin(d))
        if d[k] <= max_offset_km:
            matches.append({
                "statId": s["statId"], "name": s["statNm"], "addr": s["addr"],
                "lat": float(s["lat"]), "lng": float(s["lng"]),
                "cum_km": round(float(route_cum[k]), 1), "offset_km": round(float(d[k]), 2),
            })
    return sorted(matches, key=lambda m: m["cum_km"])


def plan_stops(matched_sorted, rate_kwh100, total_km, batt=None, soc_start=100.0,
               soc_min=20.0, soc_target=80.0):
    """그리디 SOC 시뮬레이션: 다음 구간에서 soc_min 밑으로 떨어지면 직전 휴게소에서
    80%까지 충전했다고 가정하고 이어간다 (soc_simulation.ipynb 와 같은 전략)."""
    batt = batt or TRUCK_SPEC["battery_usable_kwh"]
    waypoints = ([{"cum_km": 0.0, "name": "출발"}] + list(matched_sorted)
                + [{"cum_km": total_km, "name": "도착"}])

    soc, pos_km, recommended, i, stuck = soc_start, 0.0, [], 1, False
    while i < len(waypoints):
        leg_km = waypoints[i]["cum_km"] - pos_km
        leg_kwh = rate_kwh100 * leg_km / 100
        soc_after = soc - leg_kwh / batt * 100
        if soc_after < soc_min:
            if i == 1:  # 첫 구간부터 못 버티면 충전해도 답이 없다 — 무한루프 방지
                stuck = True
                break
            charge_stop = waypoints[i - 1]
            if charge_stop not in recommended:
                recommended.append(charge_stop)
            soc, pos_km = soc_target, charge_stop["cum_km"]
            continue
        soc, pos_km, i = soc_after, waypoints[i]["cum_km"], i + 1

    return recommended, round(soc, 1), stuck


@st.cache_data(ttl=600, show_spinner=False)
def cached_route_search(origin_q, dest_q):
    ox, oy, o_label = kakao_geocode(origin_q)
    dx, dy, d_label = kakao_geocode(dest_q)
    route = kakao_route_polyline((ox, oy), (dx, dy))
    if route is None:
        return None
    matched = match_rest_areas_to_route(route["route_df"])
    return {"o_label": o_label, "d_label": d_label, "ox": ox, "oy": oy, "dx": dx, "dy": dy,
            "total_km": route["total_km"], "duration_min": route["duration_min"], "matched": matched}


@st.cache_data(ttl=90, show_spinner=False)
def cached_status(stat_id):
    return data_go_kr_status(stat_id)


# ── 화면 ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="전기 화물차 충전소 추천", layout="wide")
st.title("전기 화물차 근처 충전소 추천")
st.caption("현재 위치를 기준으로 예측 소모전력이 가장 적은 충전소를 추천합니다.")

spec = TRUCK_SPEC
with st.container(border=True):
    st.markdown(f"**{spec['model']}** 기준 · 출처: {spec['source']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("사용가능 에너지", f"{spec['battery_usable_kwh']} kWh")
    c2.metric("최대 총중량", f"{spec['gvw_max_t']} t (국내 상한 {spec['gvw_kr_limit_t']} t)")
    c3.metric("충전", spec["charging"])
    c4.metric("20→80% / 최대주행", f"약 {spec['charge_20_80_min']}분 · {spec['range_km']}km")

st.subheader("주행 조건")
cc1, cc2, cc3 = st.columns(3)
sliders = {}
cols = [cc1, cc2, cc3]
for spec_, col in zip(CONDITION_SPECS, cols):
    sliders[spec_["key"]] = col.slider(
        f"{spec_['label']} ({spec_['unit']})", spec_["min"], spec_["max"], spec_["default"],
        step=spec_["step"],
    )
mass_kg = sliders["mass_kg"]
ambient_temp_C = sliders["ambient_temp_C"]
driving_style_index = sliders["driving_style_index"]

tab_loc, tab_route = st.tabs(["📍 현재 위치 기준 추천", "🗺️ 출발 → 도착 경로 기준 추천"])

with tab_loc:
    st.caption(
        "Streamlit은 서버 사이드 앱이라 브라우저 GPS에 직접 접근하지 못해, 주소/장소명으로 입력받습니다. "
        "후보는 전국 고속도로 휴게소 충전소(data.go.kr, kind=C0)만 대상으로 하므로 "
        "나들목 밖 일반 충전소가 섞여 나오지 않습니다."
    )
    loc_q = st.text_input("현재 위치 (주소 또는 장소명)", placeholder="예: 충북 영동군 추풍령면")
    go_loc = st.button("근처 휴게소 추천받기", type="primary")

    if go_loc:
        if not loc_q:
            st.error("현재 위치를 입력하세요.")
        else:
            try:
                lng, lat, label = cached_geocode(loc_q)
            except Exception as e:
                st.error(f"위치를 찾을 수 없습니다: {e}")
            else:
                st.session_state["recommend_loc"] = {"lat": lat, "lng": lng, "label": label}

    if "recommend_loc" in st.session_state:
        loc = st.session_state["recommend_loc"]
        with st.spinner("근처 휴게소 검색 · 실시간 혼잡도 조회 중..."):
            candidates = cached_candidates(loc["lat"], loc["lng"], k=8)
        ranked = rank_with_conditions(candidates, {
            "mass_kg": mass_kg, "ambient_temp_C": ambient_temp_C,
            "driving_style_index": driving_style_index,
        })
        st.markdown(f"**현재 위치:** {loc['label']}")

        if not ranked:
            st.warning("근처에서 고속도로 휴게소 충전소를 찾지 못했습니다 (또는 길찾기에 실패했습니다).")
        else:
            best = ranked[0]
            avail_note = (f"충전 가능 {best['available']}기" if best["available"]
                          else f"⚠ {best['congestion']}")
            st.success(
                f"★ 추천: **{best['name']}** ({best['addr']}) — "
                f"도로거리 {best['driving_km']}km · 예측 소모 {best['predicted_total_kwh']}kWh · {avail_note}"
            )
            if not best["available"]:
                st.caption("1순위 충전소가 지금 사용 가능한 커넥터가 없어도, "
                           "예측 소모전력이 가장 적어 최상단에 표시됩니다 — 아래 표에서 다른 후보를 확인하세요.")

            m = folium.Map(location=[loc["lat"], loc["lng"]], zoom_start=11, tiles="OpenStreetMap")
            folium.Marker([loc["lat"], loc["lng"]], tooltip="현재 위치",
                          icon=folium.Icon(color="black", icon="user", prefix="fa")).add_to(m)
            bounds = [[loc["lat"], loc["lng"]]]
            for i, r in enumerate(ranked):
                bounds.append([r["lat"], r["lng"]])
                available = r["available"] or 0
                color = "orange" if i == 0 else ("green" if available > 0 else "gray")
                folium.Marker(
                    [r["lat"], r["lng"]],
                    tooltip=f"{'★ 추천 · ' if i == 0 else ''}{r['name']} ({r['congestion']})",
                    popup=(f"<b>{r['name']}</b><br>{r['congestion']}<br>"
                           f"{r['driving_km']}km · {r['predicted_total_kwh']}kWh"),
                    icon=folium.Icon(color=color, icon="bolt", prefix="fa"),
                ).add_to(m)
            m.fit_bounds(bounds)
            st_folium(m, height=440, width=None, key="map_loc", returned_objects=[])

            table = pd.DataFrame(ranked)[
                ["name", "addr", "driving_km", "predicted_total_kwh", "predicted_kwh100km", "congestion"]
            ].rename(columns={"name": "휴게소", "addr": "주소", "driving_km": "도로거리(km)",
                               "predicted_total_kwh": "예측소모(kWh)",
                               "predicted_kwh100km": "전비(kWh/100km)", "congestion": "실시간 혼잡도"})
            st.dataframe(table, width="stretch", hide_index=True)
            st.caption(
                "후보 목록은 data.go.kr 전기차 충전소 정보(kind=C0 중 진짜 고속도로 시설만, 전국 670개)에서 "
                "직선거리로 가장 가까운 8곳을 뽑고, 카카오 길찾기로 실제 도로거리를, 각 충전소 ID로 "
                "실시간 상태를 다시 조회한 것입니다. 순위는 '충전 가능한 곳 우선 → 그 안에서 예측 소모전력 적은 순'입니다."
            )

with tab_route:
    st.caption(
        "출발지·도착지를 입력하면 그 경로 위에 실제로 있는 휴게소만 추리고, "
        "배터리 460kWh · 안전 SOC 20% · 목표 충전 80% 기준으로 몇 번, 어디서 충전해야 하는지 계산합니다."
    )
    rc1, rc2 = st.columns(2)
    origin_q = rc1.text_input("출발지", placeholder="예: 부산광역시청")
    dest_q = rc2.text_input("도착지", placeholder="예: 인천공항 제1여객터미널")
    go_route = st.button("경로 검색 & 정차 계획 세우기", type="primary")

    if go_route:
        if not origin_q or not dest_q:
            st.error("출발지와 도착지를 모두 입력하세요.")
        else:
            with st.spinner("경로 검색 · 휴게소 대조 중..."):
                result = cached_route_search(origin_q, dest_q)
            if result is None:
                st.error("경로를 찾지 못했습니다. 두 지점이 도로에서 너무 멀리 떨어져 있을 수 있습니다.")
            else:
                st.session_state["route_result"] = result

    if "route_result" in st.session_state:
        geo = st.session_state["route_result"]
        avg_speed = geo["total_km"] / (geo["duration_min"] / 60) if geo["duration_min"] else 80.0
        rate = predict_rate(avg_speed, mass_kg, 0.0, ambient_temp_C, driving_style_index)
        matched = geo["matched"]
        recommended, final_soc, stuck = plan_stops(matched, rate, geo["total_km"])

        st.markdown(f"**{geo['o_label']} → {geo['d_label']}**")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("총 거리", f"{geo['total_km']:.1f} km", f"{geo['duration_min']:.0f}분")
        r2.metric("평균 속도", f"{avg_speed:.1f} km/h")
        r3.metric("예측 전비", f"{rate:.1f} kWh/100km")
        r4.metric("경로상 휴게소", f"{len(matched)}곳")

        if stuck:
            st.error("첫 구간부터 배터리로 못 버팁니다 — 이 트럭 스펙으로는 무리한 경로입니다.")
        elif not recommended:
            st.success(f"충전 없이 완주 가능합니다 (도착 SOC 약 {final_soc:.0f}%).")
        else:
            st.warning(f"총 **{len(recommended)}회** 충전이 필요합니다.")
            for i, stop in enumerate(recommended):
                status = cached_status(stop["statId"])
                st.markdown(f"{i + 1}. **{stop['name']}** ({stop['cum_km']:.0f}km 지점) — {status['label']}")

        if matched:
            m = folium.Map(location=[geo["oy"], geo["ox"]], zoom_start=8, tiles="OpenStreetMap")
            folium.Marker([geo["oy"], geo["ox"]], tooltip=f"출발: {geo['o_label']}",
                          icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
            folium.Marker([geo["dy"], geo["dx"]], tooltip=f"도착: {geo['d_label']}",
                          icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")).add_to(m)
            recommended_ids = {r["statId"] for r in recommended}
            path = [[geo["oy"], geo["ox"]]]
            for s in matched:
                path.append([s["lat"], s["lng"]])
                is_rec = s["statId"] in recommended_ids
                folium.CircleMarker(
                    [s["lat"], s["lng"]], radius=8 if is_rec else 5,
                    color="#eb6834" if is_rec else "#2a78d6",
                    fill=True, fill_opacity=0.9,
                    tooltip=f"{'⚡ 충전 추천 · ' if is_rec else ''}{s['name']} ({s['cum_km']:.0f}km)",
                ).add_to(m)
            path.append([geo["dy"], geo["dx"]])
            folium.PolyLine(path, color="#2a78d6", weight=3, opacity=0.5, dash_array="6").add_to(m)
            m.fit_bounds(path)
            st_folium(m, height=440, width=None, key="map_route", returned_objects=[])

            table = pd.DataFrame(matched)
            table["충전추천"] = table["statId"].isin(recommended_ids).map({True: "⚡", False: ""})
            table = table[["충전추천", "name", "cum_km", "offset_km", "addr"]].rename(
                columns={"name": "휴게소", "cum_km": "출발지로부터(km)",
                         "offset_km": "경로에서 떨어진 거리(km)", "addr": "주소"})
            st.dataframe(table, width="stretch", hide_index=True)
        else:
            st.info("이 경로 근처(2.5km 이내)에서 찾은 고속도로 휴게소가 없습니다.")

        st.caption(
            "구간 평균속도 하나로 전체 구간의 전비를 계산합니다 — 구간별 실제 속도·경사 데이터는 "
            "없어서(고도 미포함, soc_simulation.ipynb 참고) 평지·평균속도 가정입니다."
        )

with st.expander("📊 더보기 — 머신러닝 모델 비교 결과 (ev_energy_ml_report.ipynb)"):
    if not os.path.exists(ML_RESULTS_PATH):
        st.info("models/ml_results.json 이 없습니다. ev_energy_ml_report.ipynb 06장을 먼저 실행하세요.")
    else:
        with open(ML_RESULTS_PATH, encoding="utf-8") as f:
            ml = json.load(f)

        st.caption(f"학습 데이터 {ml['n_rows']:,}건 · 최종 모델: **{ml['best_name']}**")

        results_df = pd.DataFrame(ml["results"])
        st.dataframe(results_df, width="stretch", hide_index=True)
        st.bar_chart(results_df.set_index("모델")["Test_R2"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**특성 중요도** (Random Forest · XGBoost 평균)")
            imp = pd.Series(ml["feature_importance"]).sort_values(ascending=False)
            st.bar_chart(imp)
        with col2:
            st.markdown(f"**실제값 vs 예측값 샘플** ({ml['best_name']})")
            scatter_df = pd.DataFrame(ml["scatter_sample"])
            st.scatter_chart(scatter_df, x="실제", y="예측", height=340)
