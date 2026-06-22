import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from app.telecom_churn_service import predict_churn, predict_churn_pipeline


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

    if st.button("이탈 예측하기", type="primary", use_container_width=True):
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