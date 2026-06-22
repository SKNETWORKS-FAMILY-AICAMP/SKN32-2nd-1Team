import os
import json

import streamlit as st
import streamlit.components.v1 as components
import requests

from app.next_provider_service import predict_next_provider

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost")
OLLAMA_URL = f"http://{OLLAMA_HOST}:11434/api/generate"
OLLAMA_MODEL = "exaone3.5:2.4b"

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


def _build_competitive_analysis_prompt(input_values: dict, result: dict, current_provider: str) -> str:
    """이 고객 한 명의 예측 결과를 바탕으로, 현재 통신사 입장에서의
    경쟁 대응 전략을 분석하는 프롬프트를 만듭니다.

    개인 리텐션(이 사람을 어떻게 붙잡을지)이 아니라, 통신사 간 경쟁
    포인트(요금, 결합상품 등)를 분석하도록 역할과 분석 대상을 명시합니다.
    """
    top_competitor = max(
        (name for name in result if name != current_provider),
        key=lambda name: result[name],
    )
    top_competitor_prob = result[top_competitor] * 100

    age_label = {
        1: "만 10세 미만", 2: "19세 미만", 3: "29세 미만", 4: "39세 미만",
        5: "49세 미만", 6: "59세 미만", 7: "69세 미만", 8: "70세 이상",
    }.get(input_values["age"], str(input_values["age"]))

    income_label = {
        1: "소득 없음", 2: "50만원 미만", 3: "100만원 미만", 4: "200만원 미만",
        5: "300만원 미만", 6: "400만원 미만", 7: "500만원 미만", 8: "500만원 이상",
    }.get(input_values["income"], str(input_values["income"]))

    bundled_label = {1: "가입", 2: "미가입"}.get(input_values["is_mobile_bundled"], "미확인")

    probs_text = ", ".join(f"{name} {prob*100:.1f}%" for name, prob in result.items())

    return f"""당신은 통신 3사 경쟁 전략을 분석하는 시장 분석 전문가입니다.
아래는 머신러닝 모델이 예측한 한 고객의 통신사 이동 가능성 데이터입니다.
분석 대상은 이 개별 고객이 아니라 현재 통신사({current_provider})이며,
"이 고객을 어떻게 붙잡을지"가 아니라 "{current_provider}가 경쟁사인
{top_competitor} 대비 시장에서 어떤 위치에 있고 어떤 전략을 취해야
하는지"를 분석해야 합니다.

[고객 이동 가능성 예측]
- 현재 가입 통신사: {current_provider}
- 통신사별 이동 확률: {probs_text}
- 가장 위협적인 경쟁사: {top_competitor} (이동 확률 {top_competitor_prob:.1f}%)

[참고 데이터: 이 고객의 이용 패턴]
- 연령대: {age_label}
- 소득 수준: {income_label}
- 월평균 통신비: {input_values["monthly_total_cost"]*1000:,.0f}원
- 월평균 할부금: {input_values["monthly_installment"]*1000:,.0f}원
- 결합할인 여부: {bundled_label}
- 가입 후 관측 연수: {input_values["tenure"]}년
- 과거 누적 통신사 변경 횟수: {input_values["total_changes"]}회

위 데이터를 바탕으로 다음을 분석해주세요:
1. {current_provider} 고객이 {top_competitor}로 이동할 가능성이 높게 나온 원인으로
   추정되는 요소 (요금, 결합상품, 가입 기간 등 데이터 근거 기반)
2. {top_competitor}가 {current_provider} 대비 시장에서 갖는 경쟁 우위 지점은 무엇인지
3. {current_provider}가 이 같은 고객층의 이동을 줄이기 위해 취할 수 있는
   기업 차원의 전략 2~3가지 (개별 고객 리텐션이 아니라 요금 정책, 결합상품 설계,
   마케팅 등 기업 전략 관점)

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
                chunk = json.loads(line)
                yield chunk.get("response", "")
                if chunk.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        yield "Ollama 서버에 연결할 수 없습니다. Docker 컨테이너가 실행 중인지 확인해주세요."
    except Exception as e:
        yield f"오류가 발생했습니다: {e}"


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

    # 컬럼 1: 고객 기본 정보
    with col1:
        st.markdown('<p class="block-title">고객 기본 정보</p>', unsafe_allow_html=True)

        age = st.selectbox(
            "나이 (age)",
            options=[1, 2, 3, 4, 5, 6, 7, 8],
            format_func=lambda x: {
                1: "만 10세 미만", 2: "19세 미만", 3: "29세 미만", 4: "39세 미만",
                5: "49세 미만", 6: "59세 미만", 7: "69세 미만", 8: "70세 이상",
            }[x],
            key="next_age",
        )

        income = st.selectbox(
            "소득 (income)",
            options=[1, 2, 3, 4, 5, 6, 7, 8],
            format_func=lambda x: {
                1: "소득 없음", 2: "50만원 미만", 3: "100만원 미만", 4: "200만원 미만",
                5: "300만원 미만", 6: "400만원 미만", 7: "500만원 미만", 8: "500만원 이상",
            }[x],
            key="next_income",
        )

        tenure = st.number_input(
            "가입 후 관측 연수 (tenure)",
            min_value=1, max_value=15, step=1, value=1, key="next_tenure",
            help="이 통신사를 사용한 지 몇 번째 조사 연도인지 입력하세요.",
        )

    # 컬럼 2: 현재 이용 현황
    with col2:
        st.markdown('<p class="block-title">현재 이용 현황</p>', unsafe_allow_html=True)

        provider = st.selectbox(
            "현재 가입 통신사 (provider)",
            options=[1, 2, 3],
            format_func=lambda x: {1: "SKT", 2: "KT", 3: "LG U+"}[x],
            key="next_provider_current",
            help="이 모델은 SKT/KT/LG U+ 3사 간 이동만 예측합니다.",
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

    # 컬럼 3: 통신 비용
    with col3:
        st.markdown('<p class="block-title">통신 비용</p>', unsafe_allow_html=True)

        # ⚠️ 단위 주의: 화면에서는 "원" 단위로 입력받지만(1,000원 단위로 증감),
        # 모델은 "천원" 단위를 기대하므로 input_values 구성 시 1000으로 나눕니다.
        monthly_total_cost_won = st.number_input(
            "현재 월평균 통신비 (원)",
            min_value=0, step=1000, value=0, key="next_cost",
            help="1,000원 단위로 증감합니다. 예: 70,000원을 입력하세요.",
        )

        prev_monthly_total_cost_won = st.number_input(
            "이전 조사 시점 월평균 통신비 (원)",
            min_value=0, step=1000, value=0, key="next_prev_cost",
            help="전년도(또는 이전 조사 시점) 통신비입니다. 변화율 계산에 사용됩니다.",
        )

        monthly_installment_won = st.number_input(
            "월평균 할부금 (원)",
            min_value=0, step=1000, value=0, key="next_installment",
            help="1,000원 단위로 증감합니다. 예: 30,000원을 입력하세요.",
        )

    st.markdown("---")

    if st.button("🔀 차기 통신사 예측하기", type="primary", use_container_width=True, key="next_predict_btn"):
        input_values = {
            "age": age,
            "income": income,
            # 화면은 "원" 단위로 입력받지만 모델은 "천원" 단위를 기대하므로 변환합니다.
            "monthly_total_cost": monthly_total_cost_won // 1000,
            "monthly_installment": monthly_installment_won // 1000,
            "provider": provider,
            "is_mobile_bundled": is_mobile_bundled,
            "total_changes": total_changes,
            "prev_monthly_total_cost": prev_monthly_total_cost_won // 1000,
            "tenure": tenure,
        }

        try:
            result = predict_next_provider(input_values)

            # AI 분석 버튼을 눌러 화면이 다시 실행되어도 결과가 사라지지 않도록
            # 세션에 저장합니다. 아래 렌더링 블록은 이 값을 보고 항상 그립니다.
            current_provider_name = {1: "SKT", 2: "KT", 3: "LG U+"}[provider]
            st.session_state["next_predict_input_values"] = input_values
            st.session_state["next_predict_result"] = result
            st.session_state["next_predict_current_provider"] = current_provider_name
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")

    # 예측 결과가 세션에 있으면 항상 그립니다. (AI 분석 버튼을 눌러 화면이
    # 다시 실행되어도 "차기 통신사 예측하기" 버튼은 다시 눌리지 않으므로,
    # 위 if 블록 밖에서 세션 값을 기준으로 그려야 결과가 사라지지 않습니다.)
    if "next_predict_result" in st.session_state:
        st.markdown('<p class="block-title">예측 결과</p>', unsafe_allow_html=True)
        _render_next_provider_result(st.session_state["next_predict_result"])

        st.markdown("---")
        st.markdown('<p class="block-title">🤖 AI 경쟁 전략 분석 (EXAONE 3.5)</p>', unsafe_allow_html=True)
        if st.button("경쟁사 대비 전략 AI 분석 요청", type="primary", use_container_width=True, key="next_ai_analyze_btn"):
            prompt = _build_competitive_analysis_prompt(
                st.session_state["next_predict_input_values"],
                st.session_state["next_predict_result"],
                st.session_state["next_predict_current_provider"],
            )
            with st.spinner("EXAONE이 분석 중입니다..."):
                st.write_stream(_stream_ollama(prompt))