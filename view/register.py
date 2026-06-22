import cv2
import streamlit as st

from app.face_auth import draw_face_box, read_camera_image, register_face


def render_register():
    st.subheader("회원 정보 + 얼굴 등록")

    register_user_id = st.text_input("아이디", placeholder="예: user01", key="register_user_id")
    register_password = st.text_input("암호", type="password", key="register_password")
    register_name = st.text_input("이름", placeholder="예: 홍길동", key="register_name")

    register_camera_file = st.camera_input("등록할 얼굴을 촬영하세요.", key="register_camera")
    register_image_bgr = read_camera_image(register_camera_file)

    if register_image_bgr is not None:
        annotated_bgr, face_found, face_message = draw_face_box(register_image_bgr)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption=face_message, width=420)
        if not face_found:
            st.warning(face_message)

    if st.button("회원 및 얼굴 등록 실행", type="primary"):
        if register_image_bgr is None:
            st.warning("등록할 얼굴 이미지를 카메라로 촬영하세요.")
        else:
            try:
                ok, message = register_face(
                    register_user_id,
                    register_password,
                    register_name,
                    register_image_bgr,
                )
                if ok:
                    st.success("회원 및 얼굴 등록이 완료되었습니다.")
                    st.session_state.auth_step = "login"
                    st.rerun()
                else:
                    st.error(message)
            except Exception as e:
                st.error("회원/얼굴 등록 중 오류가 발생했습니다.")
                st.exception(e)
    st.divider()
    st.caption("계정이 있으시면")
    if st.button("돌아가기"):
        st.session_state.auth_step = "login"
        st.rerun()