import os
import openai
import tiktoken

def generate(Lecture_Type, Target_Audience, API_key):
    os.environ["OPENAI_API_KEY"] = API_key

    client = openai.OpenAI()

    prompt = f"""당신은 교육팀의 마케팅 전문가입니다. 아래의 형식을 참고하여 IT 강의에 대한 홍보 문구를 자세히 작성해 주세요.
                    수강생들의 관심을 끄는 매력적인 제목을 사용하고, 홍보 문구에는 이모티콘을 자주 넣어주세요.
                    목적 : 교육 과정 제안
                    톤 & 매너: 격식 있는, 전문성을 갖춘 어조로 작성해주세요.

                    #### (제목)
                    ##### 강의 개요
                    - 수강자의 관심을 끌 수 있는 문구를 사용해, 강의의 목적과 기대효과를 설명해주세요.
                    ##### 커리큘럼
                    - 주요 학습 내용을 순서를 지정하여 핵심 포인트를 나타내고, 그에 대한 세부사항은 불릿 형태로 상세하게 나열하세요.
                    ##### 강사 소개
                    ##### 수강 혜택 및 차별점
                    ##### 신청 방법 및 문의"""


    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"강의 종류는 {Lecture_Type}이며, 수강 대상은 {Target_Audience}입니다. 이를 반영하여 맞춤형 홍보 문구를 작성해주세요."}
        ],
        temperature=0.8,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    prompt_text = response.choices[0].message.content 

    # 토큰 비용 측정하기
    encoder = tiktoken.get_encoding("cl100k_base")
    input_token = len(encoder.encode(prompt))
    output_token = len(encoder.encode(prompt_text))
    expected_sum_bill = (input_token * (0.005/1000)) + (output_token * (0.015/1000))

    return prompt_text, expected_sum_bill