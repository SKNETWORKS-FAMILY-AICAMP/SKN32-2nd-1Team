import cv2
import streamlit as st

from app.db import verify_user_password
from app.face_auth import (
    DEFAULT_SIMILARITY_THRESHOLD,
    draw_face_box,
    read_camera_image,
    verify_face_for_user,
)
from app.ui import reset_pending_face_auth


def render_login():
    st.subheader("아이디/암호 로그인 후 얼굴 2차 인증")

    login_user_id = st.text_input("아이디", key="login_user_id")
    login_password = st.text_input("암호", type="password", key="login_password")

    if st.button("1차 아이디/암호 확인", type="primary"):
        reset_pending_face_auth()

        if not login_user_id.strip() or not login_password.strip():
            st.warning("아이디와 암호를 모두 입력하세요.")
        else:
            try:
                ok, user_name, message = verify_user_password(login_user_id.strip(), login_password)
                if ok:
                    st.session_state.pending_face_user_id = login_user_id.strip()
                    st.session_state.pending_face_user_name = user_name
                    st.success(message)
                else:
                    st.error(message)
            except Exception as e:
                st.error("아이디/암호 확인 중 오류가 발생했습니다.")
                st.exception(e)

    if st.session_state.pending_face_user_id:
        st.divider()
        st.info(f"{st.session_state.pending_face_user_id} 계정의 얼굴 2차 인증을 진행하세요.")

        login_camera_file = st.camera_input("로그인할 얼굴을 촬영하세요.", key="login_camera")
        login_image_bgr = read_camera_image(login_camera_file)

        if login_image_bgr is not None:
            annotated_bgr, face_found, face_message = draw_face_box(login_image_bgr)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption=face_message, width=420)
            if not face_found:
                st.warning(face_message)

        if st.button("2차 얼굴 인증 실행", type="primary"):
            if login_image_bgr is None:
                st.warning("로그인할 얼굴 이미지를 카메라로 촬영하세요.")
            else:
                try:
                    ok, score, message = verify_face_for_user(
                        st.session_state.pending_face_user_id,
                        login_image_bgr,
                        threshold=DEFAULT_SIMILARITY_THRESHOLD,
                    )
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

    st.divider()
    st.caption("계정이 없으신가요?")
    if st.button("회원가입"):
        st.session_state.auth_step = "register"
        st.rerun()
