"""Streamlit + OpenCV + InsightFace 얼굴 로그인 + 고객 이탈 예측 메인 앱입니다."""

# pandas는 프로젝트 개요 화면의 컬럼 설명 표를 만드는 데 사용합니다.
import pandas as pd

# Streamlit은 웹 애플리케이션 화면을 구성하는 프레임워크입니다.
import streamlit as st

# DB 초기화와 아이디/암호 확인 함수를 가져옵니다.
from app.db import init_db, verify_user_password

# "개인별 이탈 예측" 메뉴는 본인이 학습시킨 xgb_pipeline.joblib 모델을 사용합니다.
# (통신사 이탈 예측 메뉴의 Pipeline 탭과 동일한 모델/변수 체계입니다.)
from app.telecom_churn_service import predict_churn_pipeline

# 세션 초기화, 로그아웃 함수를 가져옵니다.
from app.ui import init_session_state, logout

# 회원가입 / 로그인 화면을 가져옵니다.
from view.register import render_register
from view.login import render_login


# Streamlit 페이지 제목, 아이콘, 화면 폭을 설정합니다.
st.set_page_config(page_title="Face Login + Churn Prediction", page_icon="🔐", layout="wide")

# ============================================
# 개발자용: 로그인 건너뛰기 스위치
# 테스트할 때 매번 얼굴 등록/인증하기 귀찮을 때 True로 바꾸면 자동 로그인됩니다.
# 실제 발표/제출 전에는 반드시 False로 바꿔야 합니다!
# ============================================
DEV_SKIP_LOGIN = True

# 앱 시작 시 세션 상태를 초기화합니다.
init_session_state()

if "auth_step" not in st.session_state:
    st.session_state.auth_step = "login"

if DEV_SKIP_LOGIN and not st.session_state.logged_in:
    st.session_state.logged_in = True
    st.session_state.user_id = "dev_user"
    st.session_state.user_name = "테스트유저"
    st.session_state.face_score = 0.99


# MySQL DB와 사용자 테이블을 초기화합니다.
# MySQL 서버가 실행 중이어야 하며 접속 정보는 환경 변수 또는 app/db.py 기본값을 사용합니다.
if not DEV_SKIP_LOGIN:
    try:
        init_db()
    except Exception as e:
        st.error("MySQL DB 연결 또는 초기화에 실패했습니다.")
        st.exception(e)
        st.stop()

# camera_input 위젯의 화면 크기를 기본보다 작게 보이도록 CSS를 적용합니다.
# width:70%는 현재 camera_input 영역을 약 70% 크기로 줄여 표시합니다.
st.markdown(
    """
    <style>
    div[data-testid="stCameraInput"] {
        width: 70% !important;
        max-width: 520px !important;
    }
    div[data-testid="stCameraInput"] video {
        width: 100% !important;
    }
    div[data-testid="stCameraInput"] img {
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 메인 제목을 출력합니다.
st.title("고객 이탈 예측 서비스")

# 앱의 핵심 동작을 간단히 설명합니다.
st.caption("머신러닝을 활용하여 통신사 고객 이탈 확률 예측")


# 사이드바에는 슬라이더 없이 로그인 상태만 표시합니다.
with st.sidebar:
    # 사이드바 제목을 출력합니다.
    st.header("사용자 상태")

    if DEV_SKIP_LOGIN:
        st.warning("⚙️ 개발 모드: 로그인 자동 통과 중")

    # 현재 로그인 상태라면 사용자 ID와 얼굴 유사도를 표시합니다.
    if st.session_state.logged_in:
        st.success(f"로그인 사용자: {st.session_state.user_id}")
        if st.session_state.user_name:
            st.info(f"이름: {st.session_state.user_name}")
        st.info(f"얼굴 유사도: {st.session_state.face_score:.3f}")

        # 로그아웃 버튼을 누르면 세션을 초기화합니다.
        if st.button("로그아웃"):
            logout()
            st.session_state.auth_step = "register"
            st.rerun()

        st.markdown("---")
        st.header("메뉴 선택")
        menu = st.radio(
            "메뉴",
            ["프로젝트 개요", "통신사 이탈 예측"],
            label_visibility="collapsed",
        )
    else:
        # 로그인 전에는 안내 메시지를 표시합니다.
        st.warning("현재 로그인되지 않았습니다.")
        menu = None


# 로그인 전에는 얼굴 등록과 로그인 탭을 제공합니다.
if not st.session_state.logged_in:
    if st.session_state.auth_step == "register":
        render_register()
    elif st.session_state.auth_step == "login":
        render_login()

# 로그인 후에는 메뉴에 따라 화면을 표시합니다.
elif menu == "프로젝트 개요":
    import streamlit.components.v1 as components

    ACCENT = "#FF6B5B"

    components.html(f"""
    <div style="font-family:'Segoe UI', sans-serif;">
        <h1 style="font-size:1.8rem; font-weight:800; margin:0 0 4px; color:var(--text-color, inherit);">
            🚀 국내 통신사 고객 이탈 예측 프로젝트
        </h1>
    </div>
    """, height=60)

    st.markdown("### 1. 프로젝트 배경")
    st.info(
        "국내 이동통신 시장은 5G 경쟁, 결합상품 중심의 마케팅, 알뜰폰(MVNO)의 급성장 등으로 인해 "
        "고객 유치 경쟁이 매우 심화된 상태입니다. 가입자를 확보하고 유지하는 것은 통신사의 "
        "장기 수익성을 결정짓는 핵심 투자 지표입니다."
    )

    # 실제 churn_panel_v3.csv 기준 통계 (신규 변수 3개 통합 버전)
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric("총 데이터 수", "86,130건")
    with stat_col2:
        st.metric("분석 대상 기간", "2015년 ~ 2024년")
    with stat_col3:
        st.metric("이탈 데이터 수", "31,403건")

    st.caption("📌 churn_panel_v3.csv 기준 실제 통계입니다. (전체 이탈률 약 36.5%, usage_period·tablet_owned·wearable_owned 변수 통합)")

    st.markdown("---")
    st.markdown("### 2. 프로젝트 목표")
    st.markdown("""
- **주요 영향 요인 분석:** 고객 이탈에 영향을 미치는 핵심 변수 식별
- **이탈 예측 모델 개발:** 머신러닝을 활용한 최적의 예측 성능 도출
- **기업 의사결정 지원:** 데이터 기반의 마케팅 전략 및 리텐션 전략 수립 지원
""")

    st.success(
        "**핵심 가설:** 통신사 고객의 이탈은 요금 수준, 결합상품 이용 여부, 통신사 변경 이력, "
        "인구통계학적 특성 등의 복합적 요인에 의해 발생하며, 이를 통해 예측이 가능하다."
    )

    with st.expander("📊 활용 데이터 컬럼 상세 보기"):
        column_info = [
            ("pid", "개인 통합 ID", "고유 식별 번호"),
            ("year", "조사 연도", "2015 ~ 2024"),
            ("age", "나이", "1=10세미만 ~ 8=70세이상 (8단계)"),
            ("gender", "성별", "1=남성, 2=여성"),
            ("income", "개인 월평균 소득", "1=소득없음 ~ 8=500만원이상 (8단계)"),
            ("school", "최종 학력", "1=미취학 ~ 6=대학원재학이상 (6단계)"),
            ("area", "거주 지역", "코드값 1~17, 17개 시도 (라벨 미확인 — 문의 중)"),
            ("hhldsiz", "가구원 수", "코드값 1~3"),
            ("job1", "직업 유무", "1=예, 2=아니오"),
            ("mar", "결혼 여부", "1=미혼, 2=배우자있음, 3=사별, 4=이혼"),
            ("carrier", "가입 통신사", "1=SKT, 2=KT, 3=LGU+, 4=알뜰폰(MVNO)"),
            ("usage_period", "휴대폰 사용기간", "연속값 · 2021년 이후 조사 방식 기준 (단위 추가 확인 필요)"),
            ("tablet_owned", "태블릿 PC 보유 여부", "1=있다, 2=없다 · 2019년 이전 데이터는 -1(미조사)"),
            ("wearable_owned", "웨어러블 기기 보유 여부", "1=있다, 2=없다 · 2017년 이전 데이터는 -1(미조사)"),
            ("churn", "이탈 여부 (Target)", "유지(0), 이탈(1) — 전체 이탈률 약 36.5%"),
        ]
        df_columns = pd.DataFrame(column_info, columns=["컬럼명", "설명", "비고"])
        st.table(df_columns)

        st.caption(
            "📌 한국미디어패널조사 코드북(P_codebook_v32) 기준 라벨입니다. "
            "area(거주 지역)는 코드북·유저가이드에 라벨이 없어 운영기관에 문의 중입니다. "
            "usage_period·tablet_owned·wearable_owned은 결측 연도가 있던 변수를 통합해 추가했습니다."
        )

elif menu == "통신사 이탈 예측":
    from view.tab_telecom_churn import render_tab_telecom, render_tab_test_telecom

    render_tab_test_telecom()
    # tab1= st.tabs([
    #     "📊 통신사 이탈 예측(xgb_pipline)."
    # ])
    # with tab1:
        # render_tab_telecom()

