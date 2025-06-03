from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_feedback(file_path: str):
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    with open(file_path, "r", encoding="utf-8") as f:
        qa_text = f.read()

    criteria_feedback = """
    아래 항목별로 반드시 **다음과 같은 구조로** 피드백을 작성하세요. **각 항목은 반드시 고정된 제목과 순서로**, 각 항목 아래는 최소 한 문단 이상의 상세 피드백을 작성해야 합니다.

    [기술 질문 평가 항목]
    1. 핵심 기술 개념의 이해도
    평가내용 

    2. 비교 및 분석 능력
    평가내용 

    3. 성능 최적화 고려
    평가내용 

    4. 최신 트렌드 반영
    평가내용 

    5. 기술 용어의 정확성과 일관성
    평가내용 

    [포트폴리오 질문 평가 항목]
    6. 프로젝트 기반 설명력
    평가내용 

    7. 문제 해결 과정의 명확성
    평가내용 

    8. 결과 도출 및 개선 노력
    평가내용 

    9. 실무 적용 가능성
    평가내용 

    10. 협업 및 기여도 표현
    평가내용 

    ❗주의사항:
    - 위에 제시된 제목과 순서를 절대 바꾸지 마세요.
    - 각 항목 제목은 그대로 두고, 그 아래 줄부터 평가 내용을 작성하세요.
    - 불필요한 설명, 사족, 감탄사는 넣지 마세요.
    - 점수는 포함하지 마세요.
    - 반드시 한국어로 작성하세요.
    """
    feedback_prompt = ChatPromptTemplate.from_messages([
        ("system", f"당신은 시니어 소프트웨어 엔지니어이자 기술 면접관입니다. 아래의 평가 기준에 따라 지원자의 답변을 평가하세요.\n{criteria_feedback}\n"),
        ("human", "{qa_text}")
    ])
    feedback_chain = feedback_prompt | llm | StrOutputParser()
    feedback_only = feedback_chain.invoke({"qa_text": qa_text})

    with open("data/interview_feedback.txt", "w", encoding="utf-8") as f:
        f.write(feedback_only)

    return {"status": "success", "message": "interview_feedback.txt 파일로 저장 완료됨"}