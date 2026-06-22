import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from app.telecom_churn_service import (
    predict_churn_pipeline,
    predict_churn_pipeline_batch,
    PIPELINE_RAW_INPUT_COLS,
)
import io
import os
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost")
OLLAMA_URL = f"http://{OLLAMA_HOST}:11434/api/generate"
OLLAMA_MODEL = "exaone3.5:2.4b"

# ============================================================
# 디자인 색상 (app.py와 동일한 톤으로 통일)
# ============================================================
ACCENT = "#FF6B5B"
ACCENT_MID = "#E8B339"
ACCENT_LOW = "#2E9E73"

# ============================================================
# ⚠️⚠️⚠️ 매우 중요 — 절대 임의로 수정하지 마세요 ⚠️⚠️⚠️
#
# 아래 값 범위는 추측이 아니라 extracted_data.csv(147,915행)와
# 모델 학습 코드(churn_modeling.ipynb)를 직접 열어서 확인한 결과입니다.
#
#   income              : 1~8 코드값 (1=소득없음 ~ 8=500만원이상)
#   household_size      : 1~3 코드값
#   is_mobile_bundled   : 1=예, 2=아니오 (0은 데이터에 존재하지 않음)
#   provider            : 1~5 (9999=모름/무응답은 학습 데이터에 존재하지 않음)
#   school              : 1~6 (0=무학, 9999=모름/무응답은 학습 데이터에 존재하지 않음)
#   monthly_total_cost  : 천원 단위 (예: 70 = 7만원)
#   monthly_installment : 천원 단위 (예: 30 = 3만원)
#   area                : 더 이상 사용하지 않음 (모델 입력에서 제외됨)
#
# 이 부분을 고치기 전에는 반드시 extracted_data.csv와 churn_modeling.ipynb를
# 다시 검증하세요. (검증 안 하고 임의로 되돌리면 모델이 본 적 없는 값이
#  들어가서 예측이 틀어집니다.)
# ============================================================


def _build_analysis_prompt(result_df: pd.DataFrame) -> str:
    total = len(result_df)
    churn_df = result_df[result_df["prediction"] == 1]
    churn_count = len(churn_df)
    avg_prob = result_df["churn_probability"].mean() * 100

    age_map = VALUE_LABEL_MAP["age"]
    gender_map = VALUE_LABEL_MAP["gender"]
    provider_map = VALUE_LABEL_MAP["provider"]
    marriage_map = VALUE_LABEL_MAP["marriage"]

    age_dist = churn_df["age"].map(age_map).value_counts().to_dict()
    gender_dist = churn_df["gender"].map(gender_map).value_counts().to_dict()
    provider_dist = churn_df["provider"].map(provider_map).value_counts().to_dict()
    marriage_dist = churn_df["marriage"].map(marriage_map).value_counts().to_dict()
    # is_mobile_bundled: 1=예, 2=아니오 (실제 데이터 검증 결과, 0 없음)
    bundled_dist = churn_df["is_mobile_bundled"].map({1: "가입", 2: "미가입"}).value_counts().to_dict()
    # monthly_total_cost / monthly_installment는 천원 단위로 입력받으므로
    # 분석 텍스트에서는 보기 편하게 원 단위로 환산합니다.
    avg_cost = churn_df["monthly_total_cost"].mean() * 1000
    avg_installment = churn_df["monthly_installment"].mean() * 1000

    def fmt_dict(d):
        return ", ".join(f"{k} {v}명" for k, v in d.items())

    return f"""당신은 통신사 고객 이탈 분석 전문가입니다.
아래는 머신러닝 모델이 예측한 고객 이탈 위험 분석 결과입니다.

[전체 분석 요약]
- 전체 고객 수: {total}명
- 이탈 위험 고객: {churn_count}명 ({churn_count/total*100:.1f}%)
- 평균 이탈 확률: {avg_prob:.1f}%

[이탈 위험 고객 특징]
- 연령대 분포: {fmt_dict(age_dist)}
- 성별 분포: {fmt_dict(gender_dist)}
- 이동통신사: {fmt_dict(provider_dist)}
- 결혼 여부: {fmt_dict(marriage_dist)}
- 결합할인 여부: {fmt_dict(bundled_dist)}
- 평균 월 통신비: {avg_cost:,.0f}원
- 평균 월 할부금: {avg_installment:,.0f}원

위 데이터를 바탕으로 다음을 분석해주세요:
1. 이탈 위험 고객의 주요 공통 특징
2. 이탈 원인으로 추정되는 요소
3. 고객 유지를 위한 실질적인 리텐션 전략 2~3가지

한국어로 간결하게 답변해주세요."""


def _stream_ollama(prompt: str):
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                import json
                chunk = json.loads(line)
                yield chunk.get("response", "")
                if chunk.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        yield "Ollama 서버에 연결할 수 없습니다. Docker 컨테이너가 실행 중인지 확인해주세요."
    except Exception as e:
        yield f"오류가 발생했습니다: {e}"


KOREAN_COL_NAMES = {
    "year":                "조사연도",
    "age":                 "나이",
    "gender":              "성별",
    "income":              "소득",
    "school":              "학력",
    "household_size":      "가구원수",
    "job":                 "직업유무",
    "marriage":            "결혼여부",
    "provider":            "이동통신사",
    "monthly_total_cost":  "월평균통신비(천원)",
    "monthly_installment": "월평균할부금(천원)",
    "cost_payer":          "요금부담자",
    "is_mobile_bundled":   "결합할인여부",
}

VALUE_LABEL_MAP = {
    "age": {
        1: "만 10세 미만", 2: "19세 미만", 3: "29세 미만", 4: "39세 미만",
        5: "49세 미만", 6: "59세 미만", 7: "69세 미만", 8: "70세 이상",
    },
    "income": {
        1: "소득없음", 2: "50만원 미만", 3: "100만원 미만", 4: "200만원 미만",
        5: "300만원 미만", 6: "400만원 미만", 7: "500만원 미만", 8: "500만원 이상",
        9999: "모름/무응답",
    },
    "gender":   {1: "남", 2: "여"},
    "household_size": {1: "1인가구", 2: "2인가구", 3: "3인가구이상"},
    # ⚠️ 검증된 값: extracted_data.csv 실제 검증 결과 1~6만 존재 (0=무학, 9999=모름 없음).
    "school":   {1: "초등학교", 2: "중학교", 3: "중졸이하", 4: "고졸이하", 5: "대졸이하", 6: "대학원 재학 이상"},
    "job":      {1: "예", 2: "아니오"},
    "marriage": {1: "미혼", 2: "기혼", 3: "사별", 4: "이혼"},
    # ⚠️ 검증된 값: extracted_data.csv 실제 검증 결과 1~5만 존재 (9999=모름 없음).
    "provider": {1: "SKT", 2: "KT", 3: "LG U+", 4: "알뜰폰", 5: "기타(라벨 확인 필요)"},
    "cost_payer": {
        1: "본인", 2: "회사 전액부담", 3: "회사 일부지원",
        4: "가족/타인 전액부담", 5: "가족/타인 일부부담", 6: "기타",
    },
    # ⚠️ 실제 데이터 검증 결과 1=예, 2=아니오 (0 없음). 위쪽 큰 경고 주석 참고.
    "is_mobile_bundled": {1: "예", 2: "아니오"},
}


def _decode_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in VALUE_LABEL_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])
    return df


# year, id는 모델 입력에는 필요하지만 사용자가 직접 입력/확인할 값이 아니므로
# 코드에서 고정값으로 채웁니다(id는 patient_churn_service.SINGLE_PREDICT_FIXED_ID 사용).
FIXED_YEAR = 24

# 업로드 양식·미리보기·결과 화면에 노출할 컬럼 (year, id 제외).
USER_INPUT_COLS = [c for c in PIPELINE_RAW_INPUT_COLS if c not in ("year", "id")]


def _render_header(title: str, subtitle: str, height: int = 130) -> None:
    """app.py와 동일한 다크 네이비 + 코랄 그라데이션 헤더를 그립니다."""
    components.html(f"""
    <div style="
        background: linear-gradient(135deg, {ACCENT} 0%, #131B2E 100%);
        padding: 36px 40px;
        border-radius: 18px;
        font-family: 'Segoe UI', sans-serif;
        color: white;
        margin-bottom: 8px;
    ">
        <div style="display:flex; align-items:center; gap:14px;">
            <div style="
                width:52px; height:52px; border-radius:14px;
                background: rgba(255,255,255,0.15);
                display:flex; align-items:center; justify-content:center;
                font-size:26px;
            ">📡</div>
            <div>
                <h1 style="margin:0; font-size:1.8rem; font-weight:800;">
                    {title}
                </h1>
                <p style="margin:4px 0 0; opacity:0.85; font-size:0.95rem;">
                    {subtitle}
                </p>
            </div>
        </div>
    </div>
    """, height=height)


def _render_result_card(churn_prob: float) -> None:
    """app.py와 동일한 카드형 결과(좌측 보더 강조 + 진행 바)를 그립니다."""
    pct = churn_prob * 100

    if churn_prob >= 0.6:
        risk_label, risk_emoji, risk_color = "이탈 위험", "⚠️", ACCENT
    elif churn_prob >= 0.4:
        risk_label, risk_emoji, risk_color = "주의 관찰", "🟡", ACCENT_MID
    else:
        risk_label, risk_emoji, risk_color = "안정", "✅", ACCENT_LOW

    components.html(f"""
    <div style="font-family:'Segoe UI', sans-serif; display:flex; gap:16px; flex-wrap:wrap;">
        <div style="
            flex:1; min-width:220px;
            background:#131B2E; border-radius:16px; padding:24px;
            border-left:6px solid {risk_color};
            color:#F4F1EA;
        ">
            <div style="opacity:0.6; font-size:0.85rem; margin-bottom:6px;">예측 결과</div>
            <div style="font-size:1.6rem; font-weight:800;">{risk_emoji} {risk_label}</div>
        </div>
        <div style="
            flex:2; min-width:280px;
            background:#131B2E; border-radius:16px; padding:24px;
            color:#F4F1EA;
        ">
            <div style="opacity:0.6; font-size:0.85rem; margin-bottom:6px;">이탈 확률</div>
            <div style="font-size:2.4rem; font-weight:800; color:{risk_color};">
                {pct:.2f}%
            </div>
            <div style="
                margin-top:10px; height:10px; border-radius:6px;
                background:rgba(255,255,255,0.1); overflow:hidden;
            ">
                <div style="
                    width:{min(pct,100)}%; height:100%;
                    background:{risk_color}; border-radius:6px;
                    transition: width 0.4s ease;
                "></div>
            </div>
        </div>
    </div>
    """, height=160)


def render_tab_test_telecom():
    st.markdown(f"""
        <style>
        .block-title {{
            font-size: 1.2rem;
            font-weight: 700;
            border-bottom: 3px solid {ACCENT};
            padding-bottom: 6px;
            margin-bottom: 14px;
        }}
        </style>
    """, unsafe_allow_html=True)

    _render_header("개인 이탈 예측", "고객 정보를 입력하면 통신사 이탈 가능성을 예측합니다")

    col1, col2, col3 = st.columns(3)

    # 컬럼 1: 인적 사항
    with col1:
        st.markdown('<p class="block-title">인적 사항</p>', unsafe_allow_html=True)

        age = st.selectbox(
            "나이 (age)",
            # ⚠️ 검증된 값: extracted_data.csv 실제 검증 결과 1~8만 존재 (9999=모름 없음).
            options=[1, 2, 3, 4, 5, 6, 7, 8],
            format_func=lambda x: {
                1: "만 10세 미만",
                2: "19세 미만",
                3: "29세 미만",
                4: "39세 미만",
                5: "49세 미만",
                6: "59세 미만",
                7: "69세 미만",
                8: "70세 이상",
            }[x],
            key="test_age",
        )

        gender = st.selectbox(
            "성별 (gender)",
            options=[1, 2],
            format_func=lambda x: {1: "남", 2: "여"}[x],
            key="test_gender",
        )

        school = st.selectbox(
            "학력 (school)",
            # ⚠️ 검증된 값: extracted_data.csv 실제 검증 결과 1~6만 존재 (0=무학, 9999=모름 없음).
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda x: {
                1: "초등학교",
                2: "중학교",
                3: "중졸이하",
                4: "고졸이하",
                5: "대졸이하",
                6: "대학원 재학 이상",
            }[x],
            key="test_school",
        )

        marriage = st.selectbox(
            "결혼여부 (marriage)",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "미혼", 2: "기혼", 3: "사별", 4: "이혼"}[x],
            key="test_marriage",
        )

    # 컬럼 2: 가구 및 소득
    with col2:
        st.markdown('<p class="block-title">가구 및 소득</p>', unsafe_allow_html=True)

        income = st.selectbox(
            "소득 (income)",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9999],
            format_func=lambda x: {
                1: "소득없음",
                2: "50만원 미만",
                3: "100만원 미만",
                4: "200만원 미만",
                5: "300만원 미만",
                6: "400만원 미만",
                7: "500만원 미만",
                8: "500만원 이상",
                9999: "모름/무응답",
            }[x],
            key="test_income",
        )

        household_size = st.selectbox(
            "가구원수 (household_size)",
            options=[1, 2, 3],
            format_func=lambda x: {1: "1인가구", 2: "2인가구", 3: "3인가구이상"}[x],
            key="test_hhldsiz",
        )

        job = st.selectbox(
            "직업유무 (job)",
            options=[1, 2],
            format_func=lambda x: {1: "예", 2: "아니오"}[x],
            key="test_job",
        )

    # 컬럼 3: 통신 서비스 및 비용
    with col3:
        st.markdown('<p class="block-title">통신 서비스 및 비용</p>', unsafe_allow_html=True)

        provider = st.selectbox(
            "이동통신사 (provider)",
            # ⚠️ 검증된 값: extracted_data.csv 실제 검증 결과 1~5만 존재 (9999=모름 없음).
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "SKT",
                2: "KT",
                3: "LG U+",
                4: "알뜰폰",
                5: "기타(라벨 확인 필요)",
            }[x],
            key="test_provider",
        )

        # ⚠️ 단위 주의: 모델 학습 데이터(extracted_data.csv)의 monthly_total_cost,
        # monthly_installment는 "천원" 단위입니다(예: 70 = 7만원). "원" 단위가 아닙니다.
        monthly_total_cost = st.number_input(
            "월평균 통신비 (monthly_total_cost, 천원)",
            min_value=0, step=5, value=0, key="test_cost",
            help="천원 단위입니다. 예: 7만원이면 70을 입력하세요.",
        )

        monthly_installment = st.number_input(
            "월평균 할부금 (monthly_installment, 천원)",
            min_value=0, step=5, value=0, key="test_installment",
            help="천원 단위입니다. 예: 3만원이면 30을 입력하세요.",
        )

        cost_payer = st.selectbox(
            "요금 부담자 (cost_payer)",
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda x: {
                1: "본인",
                2: "회사가 전액부담",
                3: "회사가 일부지원",
                4: "가족이나 타인이 전액부담",
                5: "가족이나 타인이 일부부담",
                6: "기타",
            }[x],
            key="test_cost_payer",
        )

        # ⚠️ 검증된 값: extracted_data.csv 실제 검증 결과 1=예, 2=아니오만 존재 (0 없음).
        is_mobile_bundled = st.selectbox(
            "결합할인 여부 (is_mobile_bundled)",
            options=[1, 2],
            format_func=lambda x: {1: "예", 2: "아니오"}[x],
            key="test_bundled",
        )

    st.markdown("---")

    if st.button("이탈 예측하기", type="primary", use_container_width=True, key="test_predict_btn"):
        input_values = {
            "year": 24,
            "age": age,
            "gender": gender,
            "income": income,
            "school": school,
            "household_size": household_size,
            "job": job,
            "marriage": marriage,
            "provider": provider,
            "monthly_total_cost": monthly_total_cost,
            "monthly_installment": monthly_installment,
            "cost_payer": cost_payer,
            "is_mobile_bundled": is_mobile_bundled,
        }

        try:
            result = predict_churn_pipeline(input_values)
            _render_result_card(result["churn_probability"])
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")

    # 단일 예측 아래에 다수 고객 일괄 예측 섹션을 표시합니다.
    render_batch_prediction()


def _build_sample_csv() -> bytes:
    """업로드 양식을 안내하기 위한 샘플 CSV를 메모리에서 생성합니다."""
    # age: 1~8 (9999=모름 없음) / gender: 1=남 2=여 / provider: 1~5 (9999=모름 없음)
    # marriage: 1=미혼 2=기혼 3=사별 4=이혼 / job: 1=예 2=아니오
    # cost_payer: 1=본인 2=회사전액 3=회사일부 4=가족전액 5=가족일부 6=기타
    # school: 1~6 (0=무학, 9999=모름 없음)
    # ⚠️ income(1~8), household_size(1~3), is_mobile_bundled(1=예/2=아니오, 0 없음)는
    #    extracted_data.csv 실제 검증 결과 기준 범위입니다. 코드북 라벨은 아직 미확인입니다.
    # ⚠️ monthly_total_cost, monthly_installment는 "천원" 단위입니다(예: 70 = 7만원).
    rows = [
        {"age": 7, "gender": 1, "income": 1, "school": 6, "household_size": 1, "job": 1, "marriage": 2, "provider": 1, "monthly_total_cost": 110, "monthly_installment": 50, "cost_payer": 1, "is_mobile_bundled": 2},
        {"age": 2, "gender": 1, "income": 2, "school": 2, "household_size": 3, "job": 1, "marriage": 2, "provider": 4, "monthly_total_cost": 35, "monthly_installment": 40, "cost_payer": 5, "is_mobile_bundled": 2},
        {"age": 8, "gender": 1, "income": 3, "school": 6, "household_size": 2, "job": 2, "marriage": 2, "provider": 2, "monthly_total_cost": 55, "monthly_installment": 10, "cost_payer": 1, "is_mobile_bundled": 2},
        {"age": 2, "gender": 2, "income": 6, "school": 5, "household_size": 1, "job": 2, "marriage": 1, "provider": 4, "monthly_total_cost": 22, "monthly_installment": 50, "cost_payer": 3, "is_mobile_bundled": 2},
        {"age": 6, "gender": 1, "income": 2, "school": 1, "household_size": 2, "job": 1, "marriage": 2, "provider": 1, "monthly_total_cost": 62, "monthly_installment": 25, "cost_payer": 4, "is_mobile_bundled": 2},
        {"age": 3, "gender": 2, "income": 6, "school": 2, "household_size": 3, "job": 1, "marriage": 2, "provider": 2, "monthly_total_cost": 29, "monthly_installment": 40, "cost_payer": 4, "is_mobile_bundled": 2},
        {"age": 7, "gender": 1, "income": 6, "school": 1, "household_size": 1, "job": 2, "marriage": 4, "provider": 3, "monthly_total_cost": 22, "monthly_installment": 20, "cost_payer": 5, "is_mobile_bundled": 2},
        {"age": 3, "gender": 2, "income": 7, "school": 6, "household_size": 1, "job": 2, "marriage": 2, "provider": 2, "monthly_total_cost": 82, "monthly_installment": 50, "cost_payer": 3, "is_mobile_bundled": 2},
        {"age": 6, "gender": 2, "income": 6, "school": 2, "household_size": 3, "job": 2, "marriage": 1, "provider": 1, "monthly_total_cost": 22, "monthly_installment": 15, "cost_payer": 6, "is_mobile_bundled": 1},
        {"age": 8, "gender": 2, "income": 2, "school": 4, "household_size": 3, "job": 2, "marriage": 3, "provider": 1, "monthly_total_cost": 110, "monthly_installment": 10, "cost_payer": 6, "is_mobile_bundled": 2},
        {"age": 8, "gender": 2, "income": 2, "school": 3, "household_size": 1, "job": 2, "marriage": 1, "provider": 3, "monthly_total_cost": 82, "monthly_installment": 15, "cost_payer": 5, "is_mobile_bundled": 1},
        {"age": 8, "gender": 2, "income": 4, "school": 2, "household_size": 1, "job": 1, "marriage": 3, "provider": 4, "monthly_total_cost": 15, "monthly_installment": 10, "cost_payer": 3, "is_mobile_bundled": 2},
        {"age": 3, "gender": 1, "income": 4, "school": 5, "household_size": 1, "job": 2, "marriage": 1, "provider": 2, "monthly_total_cost": 29, "monthly_installment": 40, "cost_payer": 5, "is_mobile_bundled": 1},
        {"age": 4, "gender": 2, "income": 4, "school": 5, "household_size": 3, "job": 2, "marriage": 4, "provider": 3, "monthly_total_cost": 70, "monthly_installment": 50, "cost_payer": 4, "is_mobile_bundled": 1},
        {"age": 3, "gender": 1, "income": 2, "school": 3, "household_size": 3, "job": 1, "marriage": 2, "provider": 1, "monthly_total_cost": 22, "monthly_installment": 0, "cost_payer": 2, "is_mobile_bundled": 1},
        {"age": 2, "gender": 2, "income": 2, "school": 5, "household_size": 2, "job": 2, "marriage": 2, "provider": 2, "monthly_total_cost": 95, "monthly_installment": 40, "cost_payer": 2, "is_mobile_bundled": 2},
        {"age": 8, "gender": 2, "income": 4, "school": 1, "household_size": 3, "job": 2, "marriage": 3, "provider": 4, "monthly_total_cost": 62, "monthly_installment": 40, "cost_payer": 6, "is_mobile_bundled": 1},
        {"age": 7, "gender": 1, "income": 1, "school": 4, "household_size": 1, "job": 1, "marriage": 2, "provider": 2, "monthly_total_cost": 82, "monthly_installment": 40, "cost_payer": 2, "is_mobile_bundled": 2},
        {"age": 3, "gender": 2, "income": 8, "school": 2, "household_size": 2, "job": 1, "marriage": 1, "provider": 1, "monthly_total_cost": 22, "monthly_installment": 20, "cost_payer": 2, "is_mobile_bundled": 2},
        {"age": 5, "gender": 2, "income": 4, "school": 4, "household_size": 1, "job": 2, "marriage": 1, "provider": 4, "monthly_total_cost": 45, "monthly_installment": 40, "cost_payer": 3, "is_mobile_bundled": 2},
    ]
    for r in rows:
        r["year"] = 24
    sample = pd.DataFrame(rows, columns=USER_INPUT_COLS)
    return sample.to_csv(index=False).encode("utf-8-sig")


def _read_uploaded_table(uploaded_file) -> pd.DataFrame:
    """업로드된 파일을 확장자에 따라 CSV 또는 XLSX로 읽어 DataFrame으로 반환합니다."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        # 한글이 섞인 CSV도 안전하게 읽도록 utf-8-sig를 우선 시도합니다.
        try:
            return pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="cp949")
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("CSV 또는 XLSX 파일만 업로드할 수 있습니다.")


def render_batch_prediction():
    """CSV/XLSX 파일을 업로드받아 다수 고객의 이탈을 한 번에 예측하고 시각화합니다."""
    st.markdown("---")

    st.subheader("📁 다수 고객 일괄 예측")
    st.caption("CSV 또는 XLSX 파일을 업로드하면 모든 고객의 이탈 확률을 한 번에 예측합니다")

    # 어떤 컬럼이 필요한지 안내하고, 채워 넣을 수 있는 샘플 양식을 내려받게 합니다.
    with st.expander("📋 업로드 파일 형식 안내 / 샘플 양식 다운로드"):
        st.markdown(
            "아래 컬럼명(영문)을 헤더로 사용하고, 값은 단일 예측 화면과 동일한 코드값으로 채워주세요."
        )
        st.code(", ".join(USER_INPUT_COLS), language="text")
        st.download_button(
            "📥 샘플 CSV 양식 다운로드",
            data=_build_sample_csv(),
            file_name="telecom_churn_sample.csv",
            mime="text/csv",
            key="batch_sample_download",
        )

    uploaded = st.file_uploader(
        "고객 데이터 파일 업로드 (CSV / XLSX)",
        type=["csv", "xlsx", "xls"],
        key="batch_uploader",
    )

    if uploaded is None:
        return

    # 1) 파일 읽기
    try:
        raw_df = _read_uploaded_table(uploaded)
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    # year는 사용자가 입력하지 않으므로 코드에서 고정값으로 채워 넣습니다.
    raw_df["year"] = FIXED_YEAR

    st.markdown(f"**업로드된 행 수:** {len(raw_df)}건")
    with st.expander("업로드 데이터 미리보기 (상위 10행)"):
        # 미리보기에서는 year를 제외하고 보여줍니다.
        preview_cols = [c for c in raw_df.columns if c != "year"]
        st.dataframe(raw_df[preview_cols].head(10), use_container_width=True)

    # 2) 배치 예측
    try:
        result_df = predict_churn_pipeline_batch(raw_df)
    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {e}")
        return

    churn_count = int(result_df["prediction"].sum())
    total = len(result_df)
    retain_count = total - churn_count
    avg_prob = float(result_df["churn_probability"].mean()) * 100

    # 3) 요약 지표 - 배경 없이 텍스트 위주로 표시
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("전체 고객", f"{total:,}명")
    with m2:
        st.metric("이탈 위험", f"{churn_count:,}명")
    with m3:
        st.metric("잔류 예상", f"{retain_count:,}명")
    with m4:
        st.metric("평균 이탈 확률", f"{avg_prob:.1f}%")

    st.markdown("### 📊 예측 결과 시각화")
    viz_col1, viz_col2 = st.columns(2)

    # 3-1) 이탈/잔류 비율 도넛 차트 - app.py 색상 톤 적용
    with viz_col1:
        donut = go.Figure(
            data=[go.Pie(
                labels=["이탈 위험", "잔류 예상"],
                values=[churn_count, retain_count],
                hole=0.55,
                marker=dict(colors=[ACCENT, ACCENT_LOW]),
                textinfo="label+percent",
            )]
        )
        donut.update_layout(
            title="이탈 / 잔류 비율",
            showlegend=False,
            margin=dict(t=50, b=10, l=10, r=10),
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(donut, use_container_width=True)

    # 3-2) 이탈 확률 분포 히스토그램 - app.py 색상 톤 적용
    with viz_col2:
        hist = px.histogram(
            result_df,
            x="churn_probability",
            nbins=20,
            title="이탈 확률 분포",
            labels={"churn_probability": "이탈 확률"},
            color_discrete_sequence=[ACCENT],
        )
        # 0.5 기준선을 표시해 이탈/잔류 경계를 한눈에 보이게 합니다.
        hist.add_vline(x=0.5, line_dash="dash", line_color="#8A93A8")
        hist.update_layout(
            margin=dict(t=50, b=10, l=10, r=10),
            height=350,
            yaxis_title="고객 수",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(hist, use_container_width=True)

    # 3-3) 고위험 고객 Top N 테이블
    st.markdown("### ⚠️ 이탈 고위험 고객 (확률 높은 순)")
    top_n = min(20, total)
    high_risk = (
        result_df.sort_values("churn_probability", ascending=False)
        .head(top_n)
        .copy()
    )
    high_risk["이탈 확률(%)"] = (high_risk["churn_probability"] * 100).round(1)
    display_cols = ["이탈 확률(%)"] + USER_INPUT_COLS
    st.dataframe(
        _decode_labels(high_risk[display_cols]).reset_index(drop=True).rename(columns=KOREAN_COL_NAMES),
        use_container_width=True,
    )

    # 4) 전체 예측 결과 다운로드
    out_df = result_df.copy()
    # 다운로드 결과에서도 year, id는 제외합니다 (id는 화면에서 직접 입력받지 않는 값입니다).
    out_df = out_df.drop(columns=["year", "id"], errors="ignore")
    out_df["churn_probability"] = (out_df["churn_probability"] * 100).round(2)
    out_df["retention_probability"] = (out_df["retention_probability"] * 100).round(2)
    out_df = out_df.rename(columns={
        "churn_probability": "이탈확률(%)",
        "retention_probability": "잔류확률(%)",
        "prediction": "이탈여부(1=이탈)",
    })
    st.download_button(
        "📥 전체 예측 결과 다운로드 (CSV)",
        data=out_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="telecom_churn_predictions.csv",
        mime="text/csv",
        key="batch_result_download",
    )

    st.markdown("---")
    st.markdown('<p class="block-title">🤖 AI 분석 (EXAONE 3.5)</p>', unsafe_allow_html=True)
    if st.button("이탈 고위험 고객 AI 분석 요청", type="primary", use_container_width=True, key="ai_analyze_btn"):
        prompt = _build_analysis_prompt(result_df)
        with st.spinner("EXAONE이 분석 중입니다..."):
            st.write_stream(_stream_ollama(prompt))