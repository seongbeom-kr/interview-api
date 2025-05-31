from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_feedback(file_path: str):
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    with open(file_path, "r", encoding="utf-8") as f:
        qa_text = f.read()

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

    feedback_prompt = ChatPromptTemplate.from_messages([
        ("system", f"당신은 시니어 소프트웨어 엔지니어이자 기술 면접관입니다. 아래의 평가 기준에 따라 지원자의 답변을 평가하세요.\n{criteria_feedback}\n"),
        ("human", "{qa_text}")
    ])
    feedback_chain = feedback_prompt | llm | StrOutputParser()
    feedback_only = feedback_chain.invoke({"qa_text": qa_text})

    with open("data/interview_feedback.txt", "w", encoding="utf-8") as f:
        f.write(feedback_only)

    return {"status": "success", "message": "interview_feedback.txt 파일로 저장 완료됨"}