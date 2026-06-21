"""Streamlit + OpenCV + InsightFace 얼굴 로그인 + 고객 이탈 예측 메인 앱입니다."""

# OpenCV는 얼굴 사각형 표시 이미지의 색상 변환에 사용합니다.
import cv2

# pandas는 프로젝트 개요 화면의 컬럼 설명 표를 만드는 데 사용합니다.
import pandas as pd

# Streamlit은 웹 애플리케이션 화면을 구성하는 프레임워크입니다.
import streamlit as st

# DB 초기화와 아이디/암호 확인 함수를 가져옵니다.
from app.db import init_db, verify_user_password

# 얼굴 등록, 얼굴 검출 표시, 얼굴 2차 인증 함수를 가져옵니다.
from app.face_auth import (
    DEFAULT_SIMILARITY_THRESHOLD,
    draw_face_box,
    read_camera_image,
    register_face,
    verify_face_for_user,
)

# 고객 이탈 예측 함수를 가져옵니다.
from app.churn_service import predict_churn

# 세션 초기화, 로그아웃, 2차 인증 대기 초기화 함수를 가져옵니다.
from app.ui import init_session_state, logout, reset_pending_face_auth


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
    st.session_state.auth_step = "register"

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
            ["프로젝트 개요", "개인별 이탈 예측", "통신사 이탈 예측"],
            label_visibility="collapsed",
        )
    else:
        # 로그인 전에는 안내 메시지를 표시합니다.
        st.warning("현재 로그인되지 않았습니다.")
        menu = None


# 로그인 전에는 얼굴 등록과 로그인 탭을 제공합니다.
if not st.session_state.logged_in:
    # 얼굴 등록 탭과 로그인 탭을 생성합니다.
    # 얼굴 등록 화면
    if st.session_state.auth_step == "register":
        # 등록 섹션 제목을 출력합니다.
        st.subheader("회원 정보 + 얼굴 등록")

        # 사용자 ID를 입력받습니다.
        register_user_id = st.text_input("아이디", placeholder="예: user01", key="register_user_id")

        # 비밀번호를 입력받습니다. type=password는 화면에 비밀번호를 숨겨 표시합니다.
        register_password = st.text_input("암호", type="password", key="register_password")

        # 사용자 이름을 입력받습니다.
        register_name = st.text_input("이름", placeholder="예: 홍길동", key="register_name")

        # camera_input으로 등록 얼굴을 촬영합니다. 파일 업로더는 사용하지 않습니다.
        register_camera_file = st.camera_input("등록할 얼굴을 촬영하세요.", key="register_camera")

        # 촬영 이미지가 있으면 OpenCV 이미지로 변환합니다.
        register_image_bgr = read_camera_image(register_camera_file)

        # 촬영된 얼굴에 사각형을 표시하여 얼굴 검출 여부를 확인시킵니다.
        if register_image_bgr is not None:
            annotated_bgr, face_found, face_message = draw_face_box(register_image_bgr)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption=face_message, width=420)
            if not face_found:
                st.warning(face_message)

        # 등록 버튼을 누르면 사용자 정보와 얼굴 정보를 MySQL에 저장합니다.
        if st.button("회원 및 얼굴 등록 실행", type="primary"):
            # 이미지가 없으면 경고를 표시합니다.
            if register_image_bgr is None:
                st.warning("등록할 얼굴 이미지를 카메라로 촬영하세요.")
            else:
                try:
                    # 회원 정보와 얼굴 임베딩을 등록합니다.
                    ok, message = register_face(
                        register_user_id,
                        register_password,
                        register_name,
                        register_image_bgr,
                    )

                    # 등록 성공 시 성공 메시지를 표시합니다.
                    if ok:
                        st.success("회원 및 얼굴 등록이 완료되었습니다.")
                        st.session_state.auth_step = "login"
                        st.rerun()
                    else:
                        # 등록 실패 시 오류 메시지를 표시합니다.
                        st.error(message)
                except Exception as e:
                    st.error("회원/얼굴 등록 중 오류가 발생했습니다.")
                    st.exception(e)

    # 로그인 화면
    elif st.session_state.auth_step == "login":
        # 로그인 섹션 제목을 출력합니다.
        st.subheader("아이디/암호 로그인 후 얼굴 2차 인증")

        # 로그인할 아이디를 입력받습니다.
        login_user_id = st.text_input("아이디", key="login_user_id")

        # 로그인할 암호를 입력받습니다.
        login_password = st.text_input("암호", type="password", key="login_password")

        # 1차 인증 버튼입니다.
        if st.button("1차 아이디/암호 확인", type="primary"):
            # 기존 2차 인증 대기 상태를 초기화합니다.
            reset_pending_face_auth()

            # 아이디와 암호 입력 여부를 먼저 확인합니다.
            if not login_user_id.strip() or not login_password.strip():
                st.warning("아이디와 암호를 모두 입력하세요.")
            else:
                try:
                    # MySQL에 저장된 사용자 정보와 비밀번호 해시를 확인합니다.
                    ok, user_name, message = verify_user_password(login_user_id.strip(), login_password)

                    # 아이디/암호가 맞으면 2차 얼굴 인증 대기 상태로 전환합니다.
                    if ok:
                        st.session_state.pending_face_user_id = login_user_id.strip()
                        st.session_state.pending_face_user_name = user_name
                        st.success(message)
                    else:
                        st.error(message)
                except Exception as e:
                    st.error("아이디/암호 확인 중 오류가 발생했습니다.")
                    st.exception(e)

        # 1차 인증을 통과한 경우에만 얼굴 2차 인증 화면을 표시합니다.
        if st.session_state.pending_face_user_id:
            st.divider()
            st.info(f"{st.session_state.pending_face_user_id} 계정의 얼굴 2차 인증을 진행하세요.")

            # 로그인용 얼굴을 camera_input으로 촬영합니다. 파일 업로더는 사용하지 않습니다.
            login_camera_file = st.camera_input("로그인할 얼굴을 촬영하세요.", key="login_camera")

            # 촬영 이미지를 OpenCV BGR 이미지로 변환합니다.
            login_image_bgr = read_camera_image(login_camera_file)

            # 촬영된 얼굴에 사각형 테두리를 표시합니다.
            if login_image_bgr is not None:
                annotated_bgr, face_found, face_message = draw_face_box(login_image_bgr)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption=face_message, width=420)
                if not face_found:
                    st.warning(face_message)

            # 2차 얼굴 인증 버튼입니다.
            if st.button("2차 얼굴 인증 실행", type="primary"):
                if login_image_bgr is None:
                    st.warning("로그인할 얼굴 이미지를 카메라로 촬영하세요.")
                else:
                    try:
                        # 1차 인증을 통과한 user_id의 등록 얼굴과 현재 촬영 얼굴을 비교합니다.
                        ok, score, message = verify_face_for_user(
                            st.session_state.pending_face_user_id,
                            login_image_bgr,
                            threshold=DEFAULT_SIMILARITY_THRESHOLD,
                        )

                        # 얼굴 인증 성공 시 최종 로그인 세션을 저장합니다.
                        if ok:
                            st.session_state.logged_in = True
                            st.session_state.user_id = st.session_state.pending_face_user_id
                            st.session_state.user_name = st.session_state.pending_face_user_name
                            st.session_state.face_score = score
                            st.session_state.auth_step = "register"
                            reset_pending_face_auth()
                            st.success(f"{message} 유사도: {score:.3f}")
                            st.rerun()
                        else:
                            st.error(f"{message} 유사도: {score:.3f}")
                    except Exception as e:
                        st.error("얼굴 2차 인증 중 오류가 발생했습니다.")
                        st.exception(e)

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

    # 더미 통계 (나중에 실제 데이터로 교체)
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric("총 데이터 수", "31,572건")
    with stat_col2:
        st.metric("분석 대상 기간", "2017년 ~ 2025년")
    with stat_col3:
        st.metric("이탈 데이터 수", "11,735건")

    st.caption("⚙️ 현재는 더미 통계입니다. 데이터 전처리가 끝나면 실제 값으로 교체할 예정입니다.")

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
            ("id", "개인 통합 ID", "고유 식별 번호"),
            ("gender", "성별", "남성(0), 여성(1)"),
            ("age", "나이", "조사 연도 기준"),
            ("school", "최종 학력", "1~6 구간"),
            ("mar", "결혼 여부", "원-핫 인코딩 (mar_1~4)"),
            ("income", "개인 월평균 소득", "1~18 구간"),
            ("job", "직업 유무", "유(1), 무(0)"),
            ("region", "거주 지역", "17개 시도 원-핫 인코딩"),
            ("year", "조사 연도", "2018 ~ 2025"),
            ("phone_usage_per_m", "월평균 휴대폰 이용 요금", "만원 단위"),
            ("mobile_bundle", "휴대폰 결합상품 가입 여부", "가입(1), 미가입(0)"),
            ("telecom", "가입 통신사", "원-핫 인코딩 (skt, kt, lgu, mvno)"),
            ("telecom_change_yn", "이탈 여부 (Target)", "유지(0), 이탈(1)"),
        ]
        df_columns = pd.DataFrame(column_info, columns=["컬럼명", "설명", "비고"])
        st.table(df_columns)

# 로그인 후, "개인별 이탈 예측" 메뉴일 때 고객 이탈 예측 서비스를 표시합니다.
elif menu == "개인별 이탈 예측":
    import streamlit.components.v1 as components

    ACCENT = "#FF6B5B"
    ACCENT_MID = "#E8B339"
    ACCENT_LOW = "#2E9E73"

    # 입력 폼용 최소 CSS
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

    # 헤더 - 자유 디자인
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
            👤 {st.session_state.user_name or st.session_state.user_id}님 환영합니다
        </div>
    </div>
    """, height=170)

    # ============================================
    # 한글 표시값 ↔ 모델용 영문값 매핑
    # 화면에는 한글로 보여주고, 모델에는 학습 시 사용한 영문값을 그대로 넘깁니다.
    # ============================================
    YES_NO = {"예": "Yes", "아니오": "No"}
    GENDER_MAP = {"남성": "Male", "여성": "Female"}
    SENIOR_MAP = {"일반": "0", "고령(만 65세 이상)": "1"}
    INTERNET_MAP = {"DSL": "DSL", "광랜(Fiber optic)": "Fiber optic", "미가입": "No"}
    LINE_MAP = {"예": "Yes", "아니오": "No", "전화 미가입": "No phone service"}
    NET_OPTION_MAP = {"예": "Yes", "아니오": "No", "인터넷 미가입": "No internet service"}
    CONTRACT_MAP = {"월 단위": "Month-to-month", "1년 약정": "One year", "2년 약정": "Two year"}
    PAYMENT_MAP = {
        "자동이체(전자수표)": "Electronic check",
        "우편 청구서": "Mailed check",
        "자동이체(계좌)": "Bank transfer (automatic)",
        "자동이체(신용카드)": "Credit card (automatic)",
    }

    # 화면 표시는 원화 기준, 모델 입력은 학습 당시 사용한 달러 기준 그대로 변환합니다.
    KRW_PER_USD = 1300

    # 입력 폼을 사용하여 한 번에 고객 정보를 입력받습니다.
    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)

        # 컬럼 1: 인적 사항 + 계약 정보
        with col1:
            st.markdown('<p class="block-title">인적 사항</p>', unsafe_allow_html=True)
            gender_kr = st.selectbox("성별", list(GENDER_MAP.keys()))
            senior_kr = st.selectbox("고령 고객 여부", list(SENIOR_MAP.keys()))
            partner_kr = st.selectbox("배우자 여부", list(YES_NO.keys()))
            dependents_kr = st.selectbox("부양가족 여부", list(YES_NO.keys()))
            tenure = st.number_input("가입 기간(개월)", min_value=0, max_value=100, value=12)

        # 컬럼 2: 통신 서비스 이용 정보
        with col2:
            st.markdown('<p class="block-title">통신 이용 행태</p>', unsafe_allow_html=True)
            phone_kr = st.selectbox("전화 서비스", list(YES_NO.keys()))
            multiple_kr = st.selectbox("복수 회선", list(LINE_MAP.keys()))
            internet_kr = st.selectbox("인터넷 서비스", list(INTERNET_MAP.keys()))
            security_kr = st.selectbox("온라인 보안", list(NET_OPTION_MAP.keys()))
            backup_kr = st.selectbox("온라인 백업", list(NET_OPTION_MAP.keys()))
            protection_kr = st.selectbox("기기 보호", list(NET_OPTION_MAP.keys()))
            tech_kr = st.selectbox("기술 지원", list(NET_OPTION_MAP.keys()))
            tv_kr = st.selectbox("스트리밍 TV", list(NET_OPTION_MAP.keys()))
            movies_kr = st.selectbox("스트리밍 영화", list(NET_OPTION_MAP.keys()))

        # 컬럼 3: 결제/요금 정보
        with col3:
            st.markdown('<p class="block-title">결제 및 요금</p>', unsafe_allow_html=True)
            contract_kr = st.selectbox("계약 유형", list(CONTRACT_MAP.keys()))
            paperless_kr = st.selectbox("전자 청구서 사용", list(YES_NO.keys()))
            payment_kr = st.selectbox("결제 방식", list(PAYMENT_MAP.keys()))
            monthly_krw = st.number_input(
                "월 요금 (원)", min_value=0, max_value=390_000, value=97_000, step=1_000
            )
            total_krw = st.number_input(
                "총 요금 (원)", min_value=0, max_value=26_000_000, value=1_170_000, step=10_000
            )

        st.markdown("---")
        submitted = st.form_submit_button("🔍 이탈 여부 예측하기", type="primary")

    # 사용자가 예측 버튼을 누르면 모델 입력값을 만들고 예측을 수행합니다.
    if submitted:
        # 한글 표시값을 모델 학습 당시의 영문값으로 변환합니다.
        # 모델 입력 컬럼명은 학습 때 사용한 컬럼명과 동일해야 합니다.
        values = {
            "gender": GENDER_MAP[gender_kr],
            "SeniorCitizen": SENIOR_MAP[senior_kr],
            "Partner": YES_NO[partner_kr],
            "Dependents": YES_NO[dependents_kr],
            "tenure": tenure,
            "PhoneService": YES_NO[phone_kr],
            "MultipleLines": LINE_MAP[multiple_kr],
            "InternetService": INTERNET_MAP[internet_kr],
            "OnlineSecurity": NET_OPTION_MAP[security_kr],
            "OnlineBackup": NET_OPTION_MAP[backup_kr],
            "DeviceProtection": NET_OPTION_MAP[protection_kr],
            "TechSupport": NET_OPTION_MAP[tech_kr],
            "StreamingTV": NET_OPTION_MAP[tv_kr],
            "StreamingMovies": NET_OPTION_MAP[movies_kr],
            "Contract": CONTRACT_MAP[contract_kr],
            "PaperlessBilling": YES_NO[paperless_kr],
            "PaymentMethod": PAYMENT_MAP[payment_kr],
            # 화면에서 입력받은 원화를 모델 학습 당시 기준인 달러로 환산합니다.
            "MonthlyCharges": round(monthly_krw / KRW_PER_USD, 2),
            "TotalCharges": round(total_krw / KRW_PER_USD, 2),
        }


        # 고객 이탈 예측 서비스를 호출합니다.
        result = predict_churn(values)
        churn_prob = result["churn_probability"]
        pct = churn_prob * 100

        if churn_prob >= 0.6:
            risk_label, risk_emoji, risk_color = "이탈 위험", "⚠️", ACCENT
        elif churn_prob >= 0.4:
            risk_label, risk_emoji, risk_color = "주의 관찰", "🟡", ACCENT_MID
        else:
            risk_label, risk_emoji, risk_color = "안정", "✅", ACCENT_LOW

        # 결과 카드 - 자유 디자인
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

        # 이탈 위험이 높으면 관리 전략을 안내합니다.
        if result["prediction"] == 1:
            st.warning("이 고객은 이탈 가능성이 높습니다. 장기 계약 할인, 기술 지원 강화, 요금제 재설계를 검토하세요.")
        else:
            st.success("이 고객은 현재 잔류 가능성이 높습니다. 만족도 유지와 추가 서비스 제안을 검토하세요.")
elif menu == "통신사 이탈 예측":
    from view.tab_telecom_churn import render_tab_telecom, render_tab_test_telecom

    tab1, tab2 = st.tabs([
        "📊 통신사 이탈 예측(xgb_model)",
        "📊 통신사 이탈 예측(xgb_pipline)."
    ])
    with tab1:
        render_tab_telecom()
    with tab2:
        render_tab_test_telecom()
