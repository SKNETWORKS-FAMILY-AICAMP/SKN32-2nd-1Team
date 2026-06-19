import streamlit as st
import pandas as pd
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

# ============================================
# 커스텀 CSS - 다크 네이비 + 코랄 포인트 컬러
# ============================================
st.markdown("""
    <style>
    .stApp {
        background-color: #0F1729;
    }
    section[data-testid="stSidebar"] {
        background-color: #131B2E;
    }
    .block-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #F4F1EA;
        border-bottom: 2px solid #FF6B5B;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    .result-card {
        background-color: #131B2E;
        border: 1px solid #2A3550;
        border-radius: 12px;
        padding: 24px;
        margin-top: 12px;
    }
    .risk-high {
        border-left: 5px solid #FF6B5B;
    }
    .risk-mid {
        border-left: 5px solid #E8B339;
    }
    .risk-low {
        border-left: 5px solid #4CD6A8;
    }
    .model-note {
        font-size: 0.8rem;
        color: #8A93A8;
        background-color: #1A2238;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
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
# 사이드바 - 메뉴 (이미지 참고, 디자인 톤만 변경)
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
# 메인 화면
# ============================================
st.title("📡 개인 고객 이탈 가능성 예측")
st.markdown("고객 정보를 입력하면 모델이 이탈 확률을 계산합니다.")

st.markdown(
    '<div class="model-note">⚙️ 현재는 더미 예측 결과입니다. 모델 연결 전 화면 구조 확인용입니다.</div>',
    unsafe_allow_html=True
)

with st.expander("💰 소득 수준 가이드 확인하기"):
    st.write("1~18 단계로 구분되며, 숫자가 높을수록 소득 수준이 높습니다.")

with st.expander("🎓 학력 수준 가이드 확인하기"):
    st.write("1(미취학) ~ 6(대학원) 단계로 구분됩니다.")

st.markdown("---")

col1, col2, col3 = st.columns(3)

# ---------- 컬럼 1: 인적 사항 ----------
with col1:
    st.markdown('<p class="block-title">인적 사항</p>', unsafe_allow_html=True)

    p__age = st.number_input("나이", min_value=10, max_value=100, value=30, step=1)
    p__school = st.slider("학력 (1~6)", min_value=1, max_value=6, value=3)
    p__mar = st.selectbox("결혼 여부", ["미혼", "기혼", "이혼", "사별"])
    p__job1 = st.radio("직업 유무", ["유 (1)", "무 (0)"], horizontal=True)
    p__gender = st.radio("성별", ["남성", "여성"], horizontal=True)

# ---------- 컬럼 2: 통신 이용 행태 ----------
with col2:
    st.markdown('<p class="block-title">통신 이용 행태</p>', unsafe_allow_html=True)

    p__a02014 = st.selectbox("일반휴대폰 가입 통신사", ["SKT", "KT", "LGU+", "알뜰폰", "해당없음"])
    p__a03008 = st.selectbox("스마트폰 가입 통신사", ["SKT", "KT", "LGU+", "알뜰폰", "해당없음"])
    p__hhldsiz = st.number_input("가구원 수", min_value=1, max_value=10, value=3, step=1)

# ---------- 컬럼 3: 경제 지표 ----------
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
# 예측 결과
# ============================================
if predict_clicked:
    features = {
        "age": p__age,
        "school": p__school,
        "mar": p__mar,
        "job1": p__job1,
        "gender": p__gender,
        "a02014": p__a02014,
        "a03008": p__a03008,
        "hhldsiz": p__hhldsiz,
        "income": p__income,
        "area": p__area,
        "year": base_year,
    }

    churn_prob = predict_churn(features)

    if churn_prob >= 0.6:
        risk_label = "⚠️ 이탈 위험"
        risk_class = "risk-high"
    elif churn_prob >= 0.4:
        risk_label = "🟡 주의 관찰"
        risk_class = "risk-mid"
    else:
        risk_label = "✅ 안정"
        risk_class = "risk-low"

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.markdown(f"""
        <div class="result-card {risk_class}">
            <h4 style="margin:0; color:#8A93A8;">예측 결과</h4>
            <h2 style="margin:8px 0; color:#F4F1EA;">{risk_label}</h2>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
        <div class="result-card">
            <h4 style="margin:0; color:#8A93A8;">이탈 확률</h4>
            <h1 style="margin:8px 0; color:#FF6B5B;">{churn_prob*100:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(churn_prob, 1.0))

    st.info(
        f"💡 분석 결과: 이 고객은 현재 **{p__a03008}** 이용 중이며, "
        f"가구원 수 **{p__hhldsiz}명**, 소득 수준 **{p__income}단계** 상태입니다. "
        f"({base_year}년 기준 데이터)"
    )

else:
    st.caption("정보를 입력하고 **'이탈 여부 예측하기'** 버튼을 눌러주세요.")