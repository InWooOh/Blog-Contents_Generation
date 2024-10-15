from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.prompts import PromptTemplate
import os, re, time
import tiktoken

def generate(Lecture_Type, Target_Audience, curriculum, API_key):

    os.environ["OPENAI_API_KEY"] = API_key

    # 대주제 - 소주제 분리 과정 추가
    curriculum_list = []
    for item in curriculum:  
        curriculum_list.append(item.split('-'))

    ###########################################################################################################################################
    # 현재 파일의 절대 경로 가져오기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "Curriculum 정보.csv")
    
    # 커리큘럼 문서
    loader = CSVLoader(csv_path, csv_args={"fieldnames": ["과정명", "난이도", "대상", "분류1", "분류2", "세부 내용", "과정 분류"]}, encoding="utf-8-sig")
    data_커리큘럼 = loader.load()

    vectorstore1 = FAISS.from_documents(documents=data_커리큘럼, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))

    # BM25 리트리버 생성
    bm25_retriever = BM25Retriever.from_documents(data_커리큘럼)
    bm25_retriever.k = 1  # 상위 1개 문서 반환

    # 기존 벡터 스토어 리트리버
    faiss_retriever = vectorstore1.as_retriever(search_kwargs={"k": 1})

    # 앙상블 리트리버 생성
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

    docs_list = []  # docs를 저장할 리스트 초기화

    for item in curriculum_list:  # 입력값의 각 요소에 대해 반복
        분류2_list = item[1].split(',')  # 분류2를 ','로 분리
        for 분류2 in 분류2_list:  # 분리된 각 분류2에 대해 반복
            query = f"과정명: {Lecture_Type}\n분류2: {분류2.strip()}\n과정 분류: {item[0]}"  # 각 요소에 대해 쿼리 생성
            docs = ensemble_retriever.invoke(query)
            docs_list.extend(docs)  # 검색된 docs를 리스트에 추가

    # 세부 내용 추출 후 중복 문서는 제거
    detailed_contents = list(set(doc.page_content.split('분류1: ')[1] for doc in docs_list if '분류1: ' in doc.page_content))

    ###########################################################################################################################################
    # 현재 파일의 절대 경로 가져오기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(current_dir, "[예시 템플릿]실무 프로젝트 기반 LLM 서비스 개발자 양성과정.docx")
    
    # 예시 템플릿 문서
    loader = Docx2txtLoader(docs_path)
    data_템플릿 = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap  = 150,
        length_function = len
    )

    texts = text_splitter.split_documents(data_템플릿)

    vectorstore2 = FAISS.from_documents(documents=texts, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))

    # 교육 일정
    text_교육일정 = vectorstore2.similarity_search_with_relevance_scores("교육일정", k=1)
    교육일정 = text_교육일정[0][0].page_content

    # 신청 안내 - 신청 대상/우대 대상/신청 방법/선발 절차/면접 안내/중간 평가 안내(수료 기준)
    text_신청안내 = vectorstore2.similarity_search_with_relevance_scores("신청안내", k=2)
    신청안내 = text_신청안내[0][0].page_content + text_신청안내[1][0].page_content

    # 수료 혜택 - 인턴 공고
    text_인턴 = vectorstore2.similarity_search_with_relevance_scores("인턴", k=1)
    인턴공고 = text_인턴[0][0].page_content

    ###########################################################################################################################################
    # 현재 파일의 절대 경로 가져오기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path2 = os.path.join(current_dir, "Q&A 문서.csv")
    
    # Q&A 문서
    index_name = "blog-contents"
    os.environ['PINECONE_API_KEY'] = "b887aced-9ef7-4af5-97c0-d5c8689889e2"
    # os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")  # 환경 변수에서 API 키 가져오기
    pc = Pinecone()

    loader = CSVLoader(csv_path2, csv_args={"fieldnames": ["질문", "대답", "태그"]}, encoding='utf-8-sig')
    data_QnA = loader.load()

    # 메타데이터 추가 및 "태그: " 부분 제거
    for doc in data_QnA:
        # 태그 추출
        tag_match = re.search(r"태그: (.+)", doc.page_content)
        if tag_match:
            doc.metadata['custom_tag'] = tag_match.group(1).strip()  # 태그 내용 추가
            doc.page_content = doc.page_content.replace(tag_match.group(0), '').strip()  # "태그: " 부분 제거 


    # 파인콘 벡터 DB
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # create new index
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    # 기존 벡터스토어 초기화
    pc.delete_index(index_name)  # 기존 인덱스 삭제
    pc.create_index(  # 새로운 인덱스 생성
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


    vectorstore3 = PineconeVectorStore.from_documents(
        data_QnA, embeddings, index_name="blog-contents"
    )


    # 메타 데이터에 대한 설명 추가
    metadata_field_info = [
        AttributeInfo(
            name="custom_tag",
            description="The tag of the question. One of ['공통', '비전공자', '특화']",
            type="string"
        )
    ]

    document_content_description = "QnA of a training course"


    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    time.sleep(10)

    # 태그가 '공통' 인 문서 찾기
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore3,
        document_content_description,
        metadata_field_info,
        enable_limit=True,
        search_kwargs={"k": 11}     # 공통인 문서의 개수
    )

    qna_list_공통 = retriever.invoke("custom_tag가 '공통' 인 문서들을 모두 찾아주세요.")  
    qna_list_비전공자 = retriever.invoke(f"custom_tag가 '{Target_Audience}' 인 문서들을 모두 찾아주세요." )
    qna_list_특화 = retriever.invoke("custom_tag가 '특화' 인 문서들을 모두 찾아주세요.")  

    qna_list_공통 = retriever.invoke("custom_tag가 '공통' 인 문서들을 모두 찾아주세요.")  
    qna_list_비전공자 = retriever.invoke(f"custom_tag가 '{Target_Audience}' 인 문서들을 모두 찾아주세요." )
    qna_list_특화 = retriever.invoke("custom_tag가 '특화' 인 문서들을 모두 찾아주세요.")  

    # 세부 내용 추출
    qna_contents_공통 = [doc.page_content for doc in qna_list_공통]
    qna_contents_비전공자 = [doc.page_content for doc in qna_list_비전공자]
    qna_contents_특화 = [doc.page_content.split("대답: ")[0].strip() for doc in qna_list_특화]

    ###########################################################################################################################################
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.25)

    # 프롬프트 템플릿 생성
    prompt_template = PromptTemplate(
        input_variables=["course_name", "main_topic", "Target_Audience", "date", "apply_form", "detailed_contents", "qna_contents_비전공자", "qna_contents_특화"],
        template="""
        당신은 교육팀의 마케팅 전문가입니다. 아래의 형식을 참고하여 IT 강의에 대한 홍보 문구를 자세히 작성하세요.
        홍보 문구에는 이모티콘을 활용하세요.
        목적: 교육 과정 제안
        톤 & 매너: 격식 있는, 전문성을 갖춘 어조로 작성하세요.
        ---
        
        ### (제목)
        - 반드시 {course_name}와 {Target_Audience}만 고려하여, 관심을 끄는 매력적인 제목을 사용하세요.
        #### 강의 개요
        - 수강자의 관심을 끌 수 있는 문구를 사용해, 강의의 목적과 기대효과를 설명해주세요.
        #### 교육 일정
        - {date}을 참고해서 접수 기간, 교육 기간, 강의 시간, 강의 장소에 대해 작성하세요.
        #### 커리큘럼
        - {course_name}에 대한 주요 학습 내용을 {main_topic}을 포함하여, 반드시 Part 별로 순서를 지정해 Part 1부터 Part 5 이상의 핵심 포인트를 나타내세요.
        - 이어서 그에 대한 세부사항은 불릿 형태로 상세하게 작성하고 서술하세요.
        - 커리큘럼은 반드시 {course_name}에 대한 다양한 주요 학습 내용이 포함되어 Part 5 이상 작성해야 합니다.
        - Part 1은 "사전교육 및 OT" 이며, 반드시 {course_name}에 적절한 특강을 포함해야 합니다. 이때 최신 트렌드와 {course_name}에 맞게 특강 주제 및 제목을 구체적으로 작성하세요.
        - 마지막 Part는 "프로젝트 및 실습" 이며, 최신 트렌드와 {course_name}에 맞게 적절한 프로젝트 주제 및 제목을 구체적으로 작성하세요.
        - 주요 학습 내용 중에서 {main_topic}와 관련된 내용의 경우에만 다음 내용을 참고하여 세부사항을 작성하세요: {detailed_contents}
        #### 신청 안내
        - {apply_form}을 참고해서 신청대상, 우대대상, 수강신청방법, 선발절차, 기초지식테스트 안내, 면접안내, 중간 평가 안내에 대해 작성하세요.
        #### 수료 혜택 및 차별점
        - {intern}을 참고해서 작성하세요. 이때 인턴 주 업무와 세부 내용은 {course_name}를 고려해서 적절하게 작성하세요.
        #### Q&A
        - 반드시 {qna_contents_비전공자}의 경우 모두 작성하세요.
        - {qna_contents_특화}의 경우 질문에 대한 대답은 {course_name}과 커리큘럼에 맞게 작성하세요.
        - 질문과 대답은 각각에 대해 불릿 형태로 작성하고, 이외의 어떤 정보도 출력하지 마세요. 
        """
    )

    # LLMChain 생성
    chain = prompt_template | llm  
    # 커리큘럼 생성
    curriculum_ge = chain.invoke(  
        {
            "course_name": Lecture_Type,
            "main_topic": curriculum,
            "Target_Audience": Target_Audience,
            "date": 교육일정,
            "apply_form": 신청안내,
            "intern": 인턴공고,
            "detailed_contents": detailed_contents,
            "qna_contents_비전공자": qna_contents_비전공자,
            "qna_contents_특화": qna_contents_특화
        }
    )

    # 실제 입력 프롬프트 추출
    input_prompt = prompt_template.format(
        course_name= Lecture_Type,
        main_topic= curriculum,
        Target_Audience= Target_Audience,
        date= 교육일정,
        apply_form= 신청안내,
        intern= 인턴공고,
        detailed_contents= detailed_contents,
        qna_contents_비전공자= qna_contents_비전공자,
        qna_contents_특화= qna_contents_특화
    )

    prompt_text = curriculum_ge.content

    # qna_contents_공통의 각 항목을 마크다운 형식으로 변환
    qna_contents_공통_str = "\n\n".join(
        [f"- **질문**: {qna.split('대답: ')[0].replace('질문: ', '').strip()}\n  - 대답: {qna.split('대답: ')[1].strip()}" for qna in qna_contents_공통]
    )
    
    # curriculum.content와 qna_contents_공통 결합
    final_content = f"{curriculum_ge.content}\n\n{qna_contents_공통_str}"  # 두 내용을 결합 

    ###########################################################################################################################################
    # 토큰 비용 측정하기
    encoder = tiktoken.get_encoding("cl100k_base")
    input_token = len(encoder.encode(input_prompt))
    output_token = len(encoder.encode(prompt_text))
    expected_sum_bill = (input_token * (0.005/1000)) + (output_token * (0.015/1000))

    return final_content, expected_sum_bill
