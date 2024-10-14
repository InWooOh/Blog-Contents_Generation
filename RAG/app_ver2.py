import streamlit as st
import RAG_prompt
from datetime import datetime

# 페이지 설정 변경
st.set_page_config(
    page_title="RAG 기반 IT 강의 홍보 콘텐츠 생성",
    page_icon="🌟",
    layout="centered"
)

################## 페이지 시작 부분 ######################
st.subheader("강의 종류와 수강 대상을 입력하면 맞춤형 홍보 문구를 제작해드립니다.") #subheader

# 사이드바
with st.sidebar:
    st.header("🌟IT 강의 홍보 콘텐츠 생성하기")

    # 텍스트 형태의 input
    Lecture_Type = st.text_area("Q1. 강의 종류를 입력하세요.",
                               placeholder="ex) 파이썬을 활용한 금융 데이터 분석", 
                               max_chars=30,
                               height=100)

    Target_Audience = st.text_area("Q2. 수강 대상을 입력하세요.",
                               placeholder="ex) 대학생, 비전공자, 인턴", 
                               max_chars=30,
                               height=100)
    
    curriculum = st.text_area("Q3. 커리큘럼을 입력하세요",
                               placeholder="'Part 제목 - 세부내용' 형식으로 입력하세요. ex) 분석 역량 - 머신러닝, 데이터 시각화", 
                               max_chars=100,
                               height=140)

    curriculum_list = curriculum.split('\n')  # 줄바꿈을 기준으로 분리

    st.divider()

    API_key = st.text_input("OpenAI API Key를 입력하세요")

    # Lecture_Type, Target_Audience, API_key가 모두 입력되었는지 확인
    inputs_filled = bool(Lecture_Type.strip()) and bool(Target_Audience.strip()) and bool(API_key.strip())
    btn_submit = st.button("submit", key='submit_btn', disabled=(not inputs_filled))


# submit 버튼 onclick 이벤트
if btn_submit:
    with st.spinner("콘텐츠를 생성 중입니다. 잠시만 기다려 주세요."):
        try:
            start = datetime.now()
            prompt_result = RAG_prompt.generate(Lecture_Type, Target_Audience, curriculum_list, API_key)
            # prompt_text = prompt_result
            prompt_text = prompt_result[0]
            bill = prompt_result[1]

            st.success("홍보 콘텐츠가 정상적으로 생성되었습니다!")

            # 결과 출력
            result_container = st.container(border=True)
            result_container.write(f"생성 소요 시간은 {datetime.now() - start} 입니다.")
            result_container.write(f"예상되는 부과 비용은 ${bill} 입니다.")
            st.write(prompt_text)

        except Exception as e:  # 예외 처리
            st.error(f"⚠️ API 키가 잘못되었거나 오류가 발생했습니다! 다시 시도해주세요. 오류: {str(e)}")
