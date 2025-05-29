from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# 질문 생성 엔드포인트
@app.post("/generate-questions")
async def generate_questions(pdf: UploadFile = File(...)):
    # PDF 저장
    pdf_path = f"temp_{pdf.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())

    # PDF 로딩 및 분할
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # 프로젝트/기술스택 요약
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert IT recruiter. Summarize the main projects and technical stacks from the following portfolio text. Respond in bullet points. Only return the summary, no explanations."),
        ("human", "{input}")
    ])
    summary_chain = summary_prompt | llm | StrOutputParser()
    summaries = [summary_chain.invoke({"input": chunk.page_content}) for chunk in chunks]
    full_summary = "\n".join(summaries)

    # 고퀄리티 질문 생성 (문제 해결 질문 포함, 한국어)
    question_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior software engineer and technical interviewer. "
         "Based on the following portfolio summary, generate exactly 5 high-quality, in-depth interview questions in Korean. "
         "3 questions must be based on the candidate's projects/experience, and 2 questions must be based on their technical stack. "
         "Among the project-based questions, at least one must ask in detail about a specific problem or challenge the candidate faced during a project and how they solved or overcame it. "
         "Questions should require deep reasoning, real-world application, and critical thinking. "
         "Be creative and avoid generic questions. "
         "Label each question as [Portfolio] or [Tech Stack]. "
         "Return only the questions as a numbered list. "
         "All questions must be written in Korean."),
        ("human", "{summary}")
    ])
    question_chain = question_prompt | llm | StrOutputParser()
    questions = question_chain.invoke({"summary": full_summary})

    os.remove(pdf_path)
    return PlainTextResponse(questions)

# 피드백 생성 엔드포인트
@app.post("/generate-feedback")
async def generate_feedback(qa_file: UploadFile = File(...)):
    qa_text = (await qa_file.read()).decode("utf-8")

    criteria_feedback = """
아래의 각 평가 항목별로 지원자의 답변에 대해 매우 엄격하고 구체적으로 피드백을 작성하세요. 점수는 절대 포함하지 마세요.  
각 항목별로 한 문단 이상의 상세 피드백을 작성하세요.

[기술 질문 평가 항목]
- 핵심 기술 개념의 이해도
- 비교 및 분석 능력
- 성능 최적화 고려
- 최신 트렌드 반영
- 기술 용어의 정확성과 일관성

[포트폴리오 질문 평가 항목]
- 프로젝트 기반 설명력
- 문제 해결 과정의 명확성
- 결과 도출 및 개선 노력
- 실무 적용 가능성
- 협업 및 기여도 표현

피드백은 반드시 한국어로 작성하세요.
"""
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    feedback_prompt = ChatPromptTemplate.from_messages([
        ("system", f"당신은 시니어 소프트웨어 엔지니어이자 기술 면접관입니다. 아래의 평가 기준에 따라 지원자의 답변을 평가하세요.\n{criteria_feedback}\n"),
        ("human", "{qa_text}")
    ])
    feedback_chain = feedback_prompt | llm | StrOutputParser()
    feedback_only = feedback_chain.invoke({"qa_text": qa_text})

    return PlainTextResponse(feedback_only)

# 점수화 엔드포인트
@app.post("/score-feedback")
async def score_feedback(feedback_file: UploadFile = File(...)):
    feedback_text = (await feedback_file.read()).decode("utf-8")

    criteria_score = """
아래의 각 평가 항목별로 0~100점 만점으로 매우 엄격하게 점수를 매기세요.
아래 형식만 출력하세요(불필요한 설명, 번호, 공백 없이):

핵심 기술 개념의 이해도: X/100
비교 및 분석 능력: X/100
성능 최적화 고려: X/100
최신 트렌드 반영: X/100
기술 용어의 정확성과 일관성: X/100
프로젝트 기반 설명력: X/100
문제 해결 과정의 명확성: X/100
결과 도출 및 개선 노력: X/100
실무 적용 가능성: X/100
협업 및 기여도 표현: X/100

반드시 위 10개 항목만, '항목명: X/100' 형식으로 한 줄씩 출력하세요.  
점수만 출력하고, 다른 설명은 절대 포함하지 마세요.
"""
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    score_prompt = ChatPromptTemplate.from_messages([
        ("system", f"당신은 시니어 소프트웨어 엔지니어이자 기술 면접관입니다. 아래의 피드백을 읽고 각 항목별 점수를 산출하세요.\n{criteria_score}\n"),
        ("human", "{feedback_text}")
    ])
    score_chain = score_prompt | llm | StrOutputParser()
    scores_only = score_chain.invoke({"feedback_text": feedback_text})

    return PlainTextResponse(scores_only)