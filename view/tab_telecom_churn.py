import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from app.telecom_churn_service import (
    predict_churn,
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
    bundled_dist = churn_df["is_mobile_bundled"].map({0: "미가입", 1: "가입"}).value_counts().to_dict()
    avg_cost = churn_df["monthly_total_cost"].mean()
    avg_installment = churn_df["monthly_installment"].mean()

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
    "area":                "지역",
    "household_size":      "가구원수",
    "job":                 "직업유무",
    "marriage":            "결혼여부",
    "provider":            "이동통신사",
    "monthly_total_cost":  "월평균통신비(원)",
    "monthly_installment": "월평균할부금(원)",
    "cost_payer":          "요금부담자",
    "is_mobile_bundled":   "결합할인여부",
}

VALUE_LABEL_MAP = {
    "age": {
        1: "만 10세 미만", 2: "19세 미만", 3: "29세 미만", 4: "39세 미만",
        5: "49세 미만", 6: "59세 미만", 7: "69세 미만", 8: "70세 이상", 9999: "모름/무응답",
    },
    "gender":   {1: "남", 2: "여"},
    "school":   {0: "무학", 1: "초등학교", 2: "중학교", 3: "중졸이하", 4: "고졸이하", 5: "대졸이하", 6: "대학원 재학 이상", 9999: "모름/무응답"},
    "job":      {1: "예", 2: "아니오"},
    "marriage": {1: "미혼", 2: "기혼", 3: "사별", 4: "이혼"},
    "provider": {1: "SKT", 2: "KT", 3: "LG U+", 4: "알뜰폰", 9999: "모름/무응답"},
    "cost_payer": {
        1: "본인", 2: "회사 전액부담", 3: "회사 일부지원",
        4: "가족/타인 전액부담", 5: "가족/타인 일부부담", 6: "기타",
    },
    "is_mobile_bundled": {0: "아니오", 1: "예"},
}


def _decode_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in VALUE_LABEL_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])
    return df


# year는 모델 입력에는 필요하지만 사용자가 신경 쓸 값이 아니므로 코드에서 고정합니다.
# 단일 예측(render_tab_test_telecom)에서 쓰던 값과 동일하게 맞춥니다.
FIXED_YEAR = 24

# 업로드 양식·미리보기·결과 화면에 노출할 컬럼 (year 제외).
USER_INPUT_COLS = [c for c in PIPELINE_RAW_INPUT_COLS if c != "year"]


def render_tab_telecom():
    st.markdown("""""")
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    ">
        <h1 style="color: #e94560; margin: 0 0 0.3rem 0; font-size: 2rem;">📡 통신사 이탈 예측</h1>
        <p style="color: #a8b2d8; margin: 0; font-size: 1rem;">
            고객 정보를 입력하면 통신사 이탈 가능성을 예측합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("고객 정보 입력")

    col1, col2 = st.columns(2)


    with col1:
        age = st.selectbox(
            "나이 (age)",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9999],
            format_func=lambda x: {
                1: "만 10세 미만",
                2: "19세 미만",
                3: "29세 미만",
                4: "39세 미만",
                5: "49세 미만",
                6: "59세 미만",
                7: "69세 미만",
                8: "70세 이상",
                9999: "모름/무응답",
            }[x],
        )

        gender = st.selectbox(
            "성별 (gender)",
            options=[1, 2],
            format_func=lambda x: {1: "남", 2: "여"}[x],
        )

        income = st.number_input("월평균소득 (income)", min_value=0, step=1, value=0)

        school = st.selectbox(
            "학력 (school)",
            options=[0, 1, 2, 3, 4, 5, 6, 9999],
            format_func=lambda x: {
                0: "무학",
                1: "초등학교",
                2: "중학교",
                3: "중졸이하",
                4: "고졸이하",
                5: "대졸이하",
                6: "대학원 재학 이상",
                9999: "모름/무응답",
            }[x],
        )

        area = st.selectbox(
            "지역 (area)",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            format_func=lambda x: str(x),
        )

        hhldsiz = st.number_input("가구원수 (hhldsiz)", min_value=1, step=1, value=1)

        job1 = st.selectbox(
            "직업유무 (job1)",
            options=[1, 2],
            format_func=lambda x: {1: "예", 2: "아니오"}[x],
        )

    with col2:
        mar = st.selectbox(
            "결혼여부 (mar)",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "미혼", 2: "기혼", 3: "사별", 4: "이혼"}[x],
        )



        a02014 = st.selectbox(
            "이동통신사 (a02014)",
            options=[1, 2, 3, 4, 9999],
            format_func=lambda x: {
                1: "SKT",
                2: "KT",
                3: "LG U+",
                4: "알뜰폰",
                9999: "모름/무응답",
            }[x],
        )

        c01001 = st.number_input("통신비 (c01001, 원)", min_value=0, step=1000, value=0)

        c01003 = st.number_input("단말기 할부금 (c01003, 원)", min_value=0, step=1000, value=0)

        c02001 = st.selectbox(
            "요금제 (c02001)",
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda x: {
                1: "본인",
                2: "회사가 전액부담",
                3: "회사가 일부지원",
                4: "가족이나 타인이 전액부담",
                5: "가족이나 타인이 일부부담",
                6: "기타",
            }[x],
        )

    st.divider()

    if st.button(" 🔍 이탈 여부 예측하기", type="primary", use_container_width=True):
        input_values = {
            "age": age,
            "gender": gender,
            "income": income,
            "school": school,
            "area": area,
            "hhldsiz": hhldsiz,
            "job1": job1,
            "mar": mar,
            "a02014": a02014,
            "a03008": a02014,
            "c01001": c01001,
            "c01003": c01003,
            "c02001": c02001,
        }

        try:
            result = predict_churn(input_values)
            churn_prob = result["churn_probability"] * 100
            retain_prob = result["retention_probability"] * 100

            if result["prediction"] == 1:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #3d0000, #7b0000);
                    border: 1px solid #e94560;
                    border-radius: 12px;
                    padding: 1.5rem 2rem;
                    text-align: center;
                ">
                    <h2 style="color: #ff6b6b; margin: 0 0 0.5rem 0;">⚠️ 이탈 위험</h2>
                    <p style="color: #ffd6d6; font-size: 1.1rem; margin: 0;">
                        이탈 확률 <strong style="font-size:1.4rem;">{churn_prob:.1f}%</strong> · 잔류 확률 {retain_prob:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #003d1a, #00632b);
                    border: 1px solid #00c853;
                    border-radius: 12px;
                    padding: 1.5rem 2rem;
                    text-align: center;
                ">
                    <h2 style="color: #69f0ae; margin: 0 0 0.5rem 0;">✅ 잔류 가능성 높음</h2>
                    <p style="color: #d0ffe8; font-size: 1.1rem; margin: 0;">
                        잔류 확률 <strong style="font-size:1.4rem;">{retain_prob:.1f}%</strong> · 이탈 확률 {churn_prob:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")



def render_tab_test_telecom():
    st.markdown("""""")
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    ">
        <h1 style="color: #e94560; margin: 0 0 0.3rem 0; font-size: 2rem;">📡 통신사 이탈 예측 (Pipeline)</h1>
        <p style="color: #a8b2d8; margin: 0; font-size: 1rem;">
            고객 정보를 입력하면 통신사 이탈 가능성을 예측합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("고객 정보 입력")

    col1, col2 = st.columns(2)

    with col1:
        age = st.selectbox(
            "나이 (age)",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9999],
            format_func=lambda x: {
                1: "만 10세 미만",
                2: "19세 미만",
                3: "29세 미만",
                4: "39세 미만",
                5: "49세 미만",
                6: "59세 미만",
                7: "69세 미만",
                8: "70세 이상",
                9999: "모름/무응답",
            }[x],
            key="test_age",
        )

        gender = st.selectbox(
            "성별 (gender)",
            options=[1, 2],
            format_func=lambda x: {1: "남", 2: "여"}[x],
            key="test_gender",
        )

        income = st.number_input("소득 (income)", min_value=0, step=1, value=0, key="test_income")

        school = st.selectbox(
            "학력 (school)",
            options=[0, 1, 2, 3, 4, 5, 6, 9999],
            format_func=lambda x: {
                0: "무학",
                1: "초등학교",
                2: "중학교",
                3: "중졸이하",
                4: "고졸이하",
                5: "대졸이하",
                6: "대학원 재학 이상",
                9999: "모름/무응답",
            }[x],
            key="test_school",
        )

        area = st.selectbox(
            "지역 (area)",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            format_func=lambda x: str(x),
            key="test_area",
        )

        household_size = st.number_input("가구원수 (household_size)", min_value=1, step=1, value=1, key="test_hhldsiz")

        job = st.selectbox(
            "직업유무 (job)",
            options=[1, 2],
            format_func=lambda x: {1: "예", 2: "아니오"}[x],
            key="test_job",
        )

    with col2:
        marriage = st.selectbox(
            "결혼여부 (marriage)",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "미혼", 2: "기혼", 3: "사별", 4: "이혼"}[x],
            key="test_marriage",
        )

        provider = st.selectbox(
            "이동통신사 (provider)",
            # 실제 학습 데이터(extracted_data.csv) 검증 결과 1~5, 9999 존재. 5는 라벨 미확인.
            options=[1, 2, 3, 4, 5, 9999],
            format_func=lambda x: {
                1: "SKT",
                2: "KT",
                3: "LG U+",
                4: "알뜰폰",
                5: "기타(라벨 확인 필요)",
                9999: "모름/무응답",
            }[x],
            key="test_provider",
        )

        monthly_total_cost = st.number_input("월평균 통신비 (monthly_total_cost, 원)", min_value=0, step=1000, value=0, key="test_cost")

        monthly_installment = st.number_input("월평균 할부금 (monthly_installment, 원)", min_value=0, step=1000, value=0, key="test_installment")

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

        is_mobile_bundled = st.selectbox(
            "결합할인 여부 (is_mobile_bundled)",
            options=[1, 2],  # 실제 학습 데이터(extracted_data.csv) 검증 결과 1/2만 존재 (0 없음)
            format_func=lambda x: {1: "예", 2: "아니오"}[x],
            key="test_bundled",
        )

    st.divider()

    if st.button("이탈 예측하기", type="primary", use_container_width=True, key="test_predict_btn"):
        input_values = {
            "year": 24,
            "age": age,
            "gender": gender,
            "income": income,
            "school": school,
            "area": area,
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
            churn_prob = result["churn_probability"] * 100
            retain_prob = result["retention_probability"] * 100

            if result["prediction"] == 1:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #3d0000, #7b0000);
                    border: 1px solid #e94560;
                    border-radius: 12px;
                    padding: 1.5rem 2rem;
                    text-align: center;
                ">
                    <h2 style="color: #ff6b6b; margin: 0 0 0.5rem 0;">⚠️ 이탈 위험</h2>
                    <p style="color: #ffd6d6; font-size: 1.1rem; margin: 0;">
                        이탈 확률 <strong style="font-size:1.4rem;">{churn_prob:.1f}%</strong> · 잔류 확률 {retain_prob:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #003d1a, #00632b);
                    border: 1px solid #00c853;
                    border-radius: 12px;
                    padding: 1.5rem 2rem;
                    text-align: center;
                ">
                    <h2 style="color: #69f0ae; margin: 0 0 0.5rem 0;">✅ 잔류 가능성 높음</h2>
                    <p style="color: #d0ffe8; font-size: 1.1rem; margin: 0;">
                        잔류 확률 <strong style="font-size:1.4rem;">{retain_prob:.1f}%</strong> · 이탈 확률 {churn_prob:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
    # 단일 예측 아래에 다수 고객 일괄 예측 섹션을 표시합니다.
    render_batch_prediction()


def _build_sample_csv() -> bytes:
    """업로드 양식을 안내하기 위한 샘플 CSV를 메모리에서 생성합니다."""
    # age: 2=19세미만 3=29세미만 4=39세미만 5=49세미만 6=59세미만 7=69세미만 8=70세이상
    # gender: 1=남 2=여 / provider: 1=SKT 2=KT 3=LGU+ 4=알뜰폰
    # marriage: 1=미혼 2=기혼 3=사별 4=이혼 / job: 1=예 2=아니오
    # cost_payer: 1=본인 2=회사전액 3=회사일부 4=가족전액 5=가족일부 6=기타
    rows = [
        {"age": 2, "gender": 1, "income": 100, "school": 3, "area": 1, "household_size": 4, "job": 2, "marriage": 1, "provider": 1, "monthly_total_cost": 45000, "monthly_installment": 20000, "cost_payer": 4, "is_mobile_bundled": 0},
        {"age": 2, "gender": 2, "income": 80,  "school": 3, "area": 2, "household_size": 3, "job": 2, "marriage": 1, "provider": 2, "monthly_total_cost": 38000, "monthly_installment": 15000, "cost_payer": 4, "is_mobile_bundled": 1},
        {"age": 3, "gender": 1, "income": 250, "school": 5, "area": 3, "household_size": 2, "job": 1, "marriage": 1, "provider": 3, "monthly_total_cost": 62000, "monthly_installment": 30000, "cost_payer": 1, "is_mobile_bundled": 0},
        {"age": 3, "gender": 2, "income": 300, "school": 6, "area": 1, "household_size": 1, "job": 1, "marriage": 1, "provider": 1, "monthly_total_cost": 79000, "monthly_installment": 40000, "cost_payer": 1, "is_mobile_bundled": 1},
        {"age": 4, "gender": 1, "income": 450, "school": 5, "area": 4, "household_size": 4, "job": 1, "marriage": 2, "provider": 2, "monthly_total_cost": 95000, "monthly_installment": 35000, "cost_payer": 2, "is_mobile_bundled": 1},
        {"age": 4, "gender": 2, "income": 380, "school": 5, "area": 5, "household_size": 3, "job": 1, "marriage": 2, "provider": 1, "monthly_total_cost": 88000, "monthly_installment": 40000, "cost_payer": 1, "is_mobile_bundled": 1},
        {"age": 4, "gender": 1, "income": 200, "school": 4, "area": 6, "household_size": 5, "job": 1, "marriage": 4, "provider": 4, "monthly_total_cost": 29000, "monthly_installment": 0,     "cost_payer": 1, "is_mobile_bundled": 0},
        {"age": 5, "gender": 2, "income": 500, "school": 6, "area": 1, "household_size": 4, "job": 1, "marriage": 2, "provider": 1, "monthly_total_cost": 110000,"monthly_installment": 50000, "cost_payer": 3, "is_mobile_bundled": 1},
        {"age": 5, "gender": 1, "income": 420, "school": 5, "area": 7, "household_size": 3, "job": 1, "marriage": 2, "provider": 3, "monthly_total_cost": 82000, "monthly_installment": 30000, "cost_payer": 1, "is_mobile_bundled": 0},
        {"age": 5, "gender": 2, "income": 150, "school": 4, "area": 8, "household_size": 2, "job": 2, "marriage": 3, "provider": 4, "monthly_total_cost": 22000, "monthly_installment": 0,     "cost_payer": 5, "is_mobile_bundled": 0},
        {"age": 6, "gender": 1, "income": 320, "school": 4, "area": 2, "household_size": 2, "job": 1, "marriage": 2, "provider": 2, "monthly_total_cost": 55000, "monthly_installment": 20000, "cost_payer": 1, "is_mobile_bundled": 1},
        {"age": 6, "gender": 2, "income": 280, "school": 5, "area": 3, "household_size": 1, "job": 1, "marriage": 4, "provider": 1, "monthly_total_cost": 70000, "monthly_installment": 25000, "cost_payer": 1, "is_mobile_bundled": 0},
        {"age": 6, "gender": 1, "income": 90,  "school": 2, "area": 9, "household_size": 3, "job": 2, "marriage": 2, "provider": 4, "monthly_total_cost": 18000, "monthly_installment": 0,     "cost_payer": 4, "is_mobile_bundled": 0},
        {"age": 7, "gender": 2, "income": 200, "school": 4, "area": 4, "household_size": 2, "job": 2, "marriage": 3, "provider": 2, "monthly_total_cost": 33000, "monthly_installment": 0,     "cost_payer": 5, "is_mobile_bundled": 1},
        {"age": 7, "gender": 1, "income": 260, "school": 3, "area": 5, "household_size": 2, "job": 1, "marriage": 2, "provider": 3, "monthly_total_cost": 48000, "monthly_installment": 10000, "cost_payer": 1, "is_mobile_bundled": 0},
        {"age": 7, "gender": 2, "income": 130, "school": 1, "area": 6, "household_size": 1, "job": 2, "marriage": 3, "provider": 4, "monthly_total_cost": 15000, "monthly_installment": 0,     "cost_payer": 6, "is_mobile_bundled": 0},
        {"age": 8, "gender": 1, "income": 170, "school": 2, "area": 7, "household_size": 2, "job": 2, "marriage": 3, "provider": 4, "monthly_total_cost": 12000, "monthly_installment": 0,     "cost_payer": 5, "is_mobile_bundled": 0},
        {"age": 8, "gender": 2, "income": 110, "school": 1, "area": 8, "household_size": 1, "job": 2, "marriage": 3, "provider": 4, "monthly_total_cost": 10000, "monthly_installment": 0,     "cost_payer": 5, "is_mobile_bundled": 0},
        {"age": 8, "gender": 1, "income": 300, "school": 4, "area": 2, "household_size": 3, "job": 2, "marriage": 2, "provider": 2, "monthly_total_cost": 40000, "monthly_installment": 0,     "cost_payer": 4, "is_mobile_bundled": 1},
        {"age": 8, "gender": 2, "income": 220, "school": 3, "area": 9, "household_size": 2, "job": 2, "marriage": 2, "provider": 1, "monthly_total_cost": 35000, "monthly_installment": 0,     "cost_payer": 4, "is_mobile_bundled": 1},
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
    st.divider()
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 14px;
        margin-bottom: 1rem;
    ">
        <h2 style="color: #e94560; margin: 0 0 0.3rem 0; font-size: 1.5rem;">📁 다수 고객 일괄 예측</h2>
        <p style="color: #a8b2d8; margin: 0; font-size: 0.95rem;">
            CSV 또는 XLSX 파일을 업로드하면 모든 고객의 이탈 확률을 한 번에 예측합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

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

    # 3) 요약 지표
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("전체 고객", f"{total:,}명")
    m2.metric("이탈 위험", f"{churn_count:,}명")
    m3.metric("잔류 예상", f"{retain_count:,}명")
    m4.metric("평균 이탈 확률", f"{avg_prob:.1f}%")

    st.markdown("### 📊 예측 결과 시각화")
    viz_col1, viz_col2 = st.columns(2)

    # 3-1) 이탈/잔류 비율 도넛 차트
    with viz_col1:
        donut = go.Figure(
            data=[go.Pie(
                labels=["이탈 위험", "잔류 예상"],
                values=[churn_count, retain_count],
                hole=0.55,
                marker=dict(colors=["#e94560", "#00c853"]),
                textinfo="label+percent",
            )]
        )
        donut.update_layout(
            title="이탈 / 잔류 비율",
            showlegend=False,
            margin=dict(t=50, b=10, l=10, r=10),
            height=350,
        )
        st.plotly_chart(donut, use_container_width=True)

    # 3-2) 이탈 확률 분포 히스토그램
    with viz_col2:
        hist = px.histogram(
            result_df,
            x="churn_probability",
            nbins=20,
            title="이탈 확률 분포",
            labels={"churn_probability": "이탈 확률"},
            color_discrete_sequence=["#e94560"],
        )
        # 0.5 기준선을 표시해 이탈/잔류 경계를 한눈에 보이게 합니다.
        hist.add_vline(x=0.5, line_dash="dash", line_color="#a8b2d8")
        hist.update_layout(
            margin=dict(t=50, b=10, l=10, r=10),
            height=350,
            yaxis_title="고객 수",
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
    # 다운로드 결과에서도 year는 제외합니다.
    out_df = out_df.drop(columns=["year"], errors="ignore")
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

    st.divider()
    st.markdown("### 🤖 AI 분석 (EXAONE 3.5)")
    if st.button("이탈 고위험 고객 AI 분석 요청", type="primary", use_container_width=True, key="ai_analyze_btn"):
        prompt = _build_analysis_prompt(result_df)
        with st.spinner("EXAONE이 분석 중입니다..."):
            st.write_stream(_stream_ollama(prompt))