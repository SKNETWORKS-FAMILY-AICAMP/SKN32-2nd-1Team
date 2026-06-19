import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from datetime import datetime

# ============================================
# 페이지 설정
# ============================================
st.set_page_config(
    page_title="통신사 이탈 분석 서비스",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

ACCENT = "#FF6B5B"
ACCENT_MID = "#E8B339"
ACCENT_LOW = "#2E9E73"

# ============================================
# 입력 폼용 최소 CSS (Streamlit 위젯 영역)
# ============================================
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

# ============================================
# 더미 모델 (나중에 학습된 모델로 교체할 자리)
# ============================================
def predict_churn(features: dict) -> float:
    """
    임시 더미 예측 함수.
    나중에 여기를 실제 학습된 모델(XGBoost 등) 예측 코드로 교체하면 됨.

    예: model = joblib.load("model.pkl")
        return model.predict_proba(X)[0][1]
    """
    rng = np.random.default_rng(seed=hash(str(features)) % (2**32))
    base = rng.uniform(0.15, 0.9)
    return float(base)


# ============================================
# 사이드바
# ============================================
st.sidebar.markdown("### 📡 통신사 이탈 분석 서비스")
menu = st.sidebar.radio(
    "메뉴 선택",
    ["개인별 이탈 예측", "프로젝트 개요"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.caption(f"마지막 갱신: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ============================================
# 헤더 - components.html로 자유 디자인
# ============================================
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
                개인 고객 이탈 가능성 예측
            </h1>
            <p style="margin:4px 0 0; opacity:0.85; font-size:0.95rem;">
                고객 정보를 입력하면 모델이 이탈 확률을 계산합니다
            </p>
        </div>
    </div>
    <div style="
        margin-top:18px; display:inline-block;
        background: rgba(0,0,0,0.25);
        padding: 6px 14px; border-radius: 20px;
        font-size: 0.78rem;
    ">
        ⚙️ 현재는 더미 예측 결과입니다 · 모델 연결 전 화면 구조 확인용
    </div>
</div>
""", height=170)

with st.expander("💰 소득 수준 가이드 확인하기"):
    st.write("1~18 단계로 구분되며, 숫자가 높을수록 소득 수준이 높습니다.")

with st.expander("🎓 학력 수준 가이드 확인하기"):
    st.write("1(미취학) ~ 6(대학원) 단계로 구분됩니다.")

st.markdown("---")

# ============================================
# 입력 폼 (Streamlit 위젯 - 그대로 유지)
# ============================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<p class="block-title">인적 사항</p>', unsafe_allow_html=True)
    p__age = st.number_input("나이", min_value=10, max_value=100, value=30, step=1)
    p__school = st.slider("학력 (1~6)", min_value=1, max_value=6, value=3)
    p__mar = st.selectbox("결혼 여부", ["미혼", "기혼", "이혼", "사별"])
    p__job1 = st.radio("직업 유무", ["유 (1)", "무 (0)"], horizontal=True)
    p__gender = st.radio("성별", ["남성", "여성"], horizontal=True)

with col2:
    st.markdown('<p class="block-title">통신 이용 행태</p>', unsafe_allow_html=True)
    p__a02014 = st.selectbox("일반휴대폰 가입 통신사", ["SKT", "KT", "LGU+", "알뜰폰", "해당없음"])
    p__a03008 = st.selectbox("스마트폰 가입 통신사", ["SKT", "KT", "LGU+", "알뜰폰", "해당없음"])
    p__hhldsiz = st.number_input("가구원 수", min_value=1, max_value=10, value=3, step=1)

with col3:
    st.markdown('<p class="block-title">경제 지표</p>', unsafe_allow_html=True)
    p__income = st.slider("월평균 소득 수준 (1~18)", min_value=1, max_value=18, value=5)
    base_year = st.selectbox("기준 연도", list(range(2025, 2014, -1)))
    p__area = st.selectbox(
        "거주 지역", ["서울", "경기", "인천", "강원", "충청", "전라", "경상", "제주"]
    )

st.markdown("---")
predict_clicked = st.button("🔍 이탈 여부 예측하기", use_container_width=False)

# ============================================
# 예측 결과 - components.html로 자유 디자인
# ============================================
if predict_clicked:
    features = {
        "age": p__age, "school": p__school, "mar": p__mar, "job1": p__job1,
        "gender": p__gender, "a02014": p__a02014, "a03008": p__a03008,
        "hhldsiz": p__hhldsiz, "income": p__income, "area": p__area, "year": base_year,
    }
    churn_prob = predict_churn(features)

    if churn_prob >= 0.6:
        risk_label, risk_emoji, risk_color = "이탈 위험", "⚠️", ACCENT
    elif churn_prob >= 0.4:
        risk_label, risk_emoji, risk_color = "주의 관찰", "🟡", ACCENT_MID
    else:
        risk_label, risk_emoji, risk_color = "안정", "✅", ACCENT_LOW

    pct = churn_prob * 100

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

    st.info(
        f"💡 분석 결과: 이 고객은 현재 **{p__a03008}** 이용 중이며, "
        f"가구원 수 **{p__hhldsiz}명**, 소득 수준 **{p__income}단계** 상태입니다. "
        f"({base_year}년 기준 데이터)"
    )
else:
    st.caption("정보를 입력하고 **'이탈 여부 예측하기'** 버튼을 눌러주세요.")