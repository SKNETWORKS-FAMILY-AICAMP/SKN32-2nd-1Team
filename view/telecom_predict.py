import streamlit as st
import streamlit.components.v1 as components

from app.next_provider_service import predict_next_provider

# ============================================================
# 디자인 색상 (tab_telecom_churn.py / app.py와 동일한 톤으로 통일)
# ============================================================
ACCENT = "#FF6B5B"
ACCENT_KT = "#E8B339"
ACCENT_LGU = "#2E9E73"

PROVIDER_COLORS = {"SKT": ACCENT, "KT": ACCENT_KT, "LG U+": ACCENT_LGU}

# ============================================================
# ⚠️⚠️⚠️ 매우 중요 — 절대 임의로 수정하지 마세요 ⚠️⚠️⚠️
#
# 이 화면이 사용하는 모델(next_lgb_churn_model.joblib)은 "통신사 이탈
# 예측" 화면의 모델(xgb_full_pipeline.joblib)과 완전히 다른 모델입니다.
#
# 이 모델은 "이미 통신사를 변경하기로 한(또는 변경한) 고객이 SKT/KT/
# LG U+ 중 어디로 이동할지"를 예측합니다. 즉 결과는 "이 고객이 이탈할
# 확률"이 아니라 "이 고객이 통신사를 바꾼다면 어디로 갈 가능성이
# 높은가"를 의미합니다.
#
# area는 화면에서 입력받지 않고 app/next_provider_service.py의
# AREA_FIXED_VALUE로 고정해 모델에 전달합니다 (변수 중요도가 낮아
# 화면 단순화를 위해 제외하기로 결정함).
# ============================================================


def _render_header(title: str, subtitle: str, height: int = 130) -> None:
    """app.py / tab_telecom_churn.py와 동일한 다크 네이비 + 코랄 그라데이션 헤더를 그립니다."""
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
            ">🔀</div>
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


def _render_next_provider_result(result: dict) -> None:
    """SKT/KT/LG U+ 이동 확률을 카드형으로 그립니다."""
    # 확률이 가장 높은 통신사를 강조합니다.
    top_provider = max(result, key=result.get)

    cards_html = ""
    for name, prob in result.items():
        pct = prob * 100
        color = PROVIDER_COLORS.get(name, ACCENT)
        is_top = name == top_provider and prob > 0
        border_width = "6px" if is_top else "2px"
        opacity = "1" if is_top else "0.7"

        cards_html += f"""
        <div style="
            flex:1; min-width:160px;
            background:#131B2E; border-radius:16px; padding:22px;
            border-left:{border_width} solid {color};
            color:#F4F1EA; opacity:{opacity};
        ">
            <div style="opacity:0.6; font-size:0.85rem; margin-bottom:6px;">{name}</div>
            <div style="font-size:2rem; font-weight:800; color:{color};">
                {pct:.1f}%
            </div>
            <div style="
                margin-top:10px; height:8px; border-radius:6px;
                background:rgba(255,255,255,0.1); overflow:hidden;
            ">
                <div style="
                    width:{min(pct,100)}%; height:100%;
                    background:{color}; border-radius:6px;
                    transition: width 0.4s ease;
                "></div>
            </div>
        </div>
        """

    components.html(f"""
    <div style="font-family:'Segoe UI', sans-serif; display:flex; gap:14px; flex-wrap:wrap;">
        {cards_html}
    </div>
    """, height=180)


def render_tab_next_provider():
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

    _render_header(
        "이탈 시 차기 통신사 예측",
        "통신사를 변경한다면 SKT / KT / LG U+ 중 어디로 이동할 가능성이 높은지 예측합니다",
    )

    st.caption(
        "⚠️ 이 화면은 \"이탈 여부\"가 아니라, 고객이 통신사를 변경한다고 가정했을 때 "
        "어느 통신사로 이동할 가능성이 높은지를 보여줍니다."
    )

    col1, col2, col3 = st.columns(3)

    # 컬럼 1: 인적 사항
    with col1:
        st.markdown('<p class="block-title">인적 사항</p>', unsafe_allow_html=True)

        age = st.selectbox(
            "나이 (age)",
            options=[1, 2, 3, 4, 5, 6, 7, 8],
            format_func=lambda x: {
                1: "만 10세 미만", 2: "19세 미만", 3: "29세 미만", 4: "39세 미만",
                5: "49세 미만", 6: "59세 미만", 7: "69세 미만", 8: "70세 이상",
            }[x],
            key="next_age",
        )

        gender = st.selectbox(
            "성별 (gender)",
            options=[1, 2],
            format_func=lambda x: {1: "남", 2: "여"}[x],
            key="next_gender",
        )

        school = st.selectbox(
            "학력 (school)",
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda x: {
                1: "초등학교", 2: "중학교", 3: "중졸이하",
                4: "고졸이하", 5: "대졸이하", 6: "대학원 재학 이상",
            }[x],
            key="next_school",
        )

        marriage = st.selectbox(
            "결혼여부 (marriage)",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "미혼", 2: "기혼", 3: "사별", 4: "이혼"}[x],
            key="next_marriage",
        )

    # 컬럼 2: 가구 및 소득
    with col2:
        st.markdown('<p class="block-title">가구 및 소득</p>', unsafe_allow_html=True)

        income = st.number_input(
            "소득 (income, 코드값 1~8)",
            min_value=1, max_value=8, step=1, value=1, key="next_income",
            help="1=소득없음 ~ 8=500만원이상",
        )

        household_size = st.number_input(
            "가구원수 (household_size, 코드값 1~3)",
            min_value=1, max_value=3, step=1, value=1, key="next_household_size",
        )

        job = st.selectbox(
            "직업유무 (job)",
            options=[1, 2],
            format_func=lambda x: {1: "예", 2: "아니오"}[x],
            key="next_job",
        )

        tenure = st.number_input(
            "가입 후 관측 연수 (tenure)",
            min_value=1, max_value=15, step=1, value=1, key="next_tenure",
            help="이 통신사를 사용한 지 몇 번째 조사 연도인지 입력하세요.",
        )

    # 컬럼 3: 통신 서비스 및 비용
    with col3:
        st.markdown('<p class="block-title">통신 서비스 및 비용</p>', unsafe_allow_html=True)

        provider = st.selectbox(
            "현재 가입 통신사 (provider)",
            options=[1, 2, 3],
            format_func=lambda x: {1: "SKT", 2: "KT", 3: "LG U+"}[x],
            key="next_provider_current",
            help="이 모델은 SKT/KT/LG U+ 3사 간 이동만 예측합니다.",
        )

        monthly_total_cost = st.number_input(
            "현재 월평균 통신비 (천원)",
            min_value=0, step=5, value=50, key="next_cost",
            help="천원 단위입니다. 예: 7만원이면 70을 입력하세요.",
        )

        prev_monthly_total_cost = st.number_input(
            "이전 조사 시점 월평균 통신비 (천원)",
            min_value=0, step=5, value=50, key="next_prev_cost",
            help="전년도(또는 이전 조사 시점) 통신비입니다. 변화율 계산에 사용됩니다.",
        )

        monthly_installment = st.number_input(
            "월평균 할부금 (천원)",
            min_value=0, step=5, value=0, key="next_installment",
        )

        cost_payer = st.selectbox(
            "요금 부담자 (cost_payer)",
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda x: {
                1: "본인", 2: "회사가 전액부담", 3: "회사가 일부지원",
                4: "가족이나 타인이 전액부담", 5: "가족이나 타인이 일부부담", 6: "기타",
            }[x],
            key="next_cost_payer",
        )

        is_mobile_bundled = st.selectbox(
            "결합할인 여부 (is_mobile_bundled)",
            options=[1, 2],
            format_func=lambda x: {1: "예", 2: "아니오"}[x],
            key="next_bundled",
        )

        total_changes = st.number_input(
            "과거 누적 통신사 변경 횟수 (total_changes)",
            min_value=0, max_value=10, step=1, value=1, key="next_total_changes",
        )

    st.markdown("---")

    if st.button("🔀 차기 통신사 예측하기", type="primary", use_container_width=True, key="next_predict_btn"):
        input_values = {
            "age": age,
            "gender": gender,
            "income": income,
            "school": school,
            "job": job,
            "marriage": marriage,
            "monthly_total_cost": monthly_total_cost,
            "monthly_installment": monthly_installment,
            "cost_payer": cost_payer,
            "provider": provider,
            "household_size": household_size,
            "is_mobile_bundled": is_mobile_bundled,
            "total_changes": total_changes,
            "prev_monthly_total_cost": prev_monthly_total_cost,
            "tenure": tenure,
        }

        try:
            result = predict_next_provider(input_values)
            st.markdown('<p class="block-title">예측 결과</p>', unsafe_allow_html=True)
            _render_next_provider_result(result)
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")