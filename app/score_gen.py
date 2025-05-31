from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_score(file_path: str):
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    with open(file_path, "r", encoding="utf-8") as f:
        feedback_text = f.read()

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
    """

    score_prompt = ChatPromptTemplate.from_messages([
        ("system", f"당신은 시니어 소프트웨어 엔지니어이자 기술 면접관입니다. 아래의 피드백을 읽고 각 항목별 점수를 산출하세요.\n{criteria_score}\n"),
        ("human", "{feedback_text}")
    ])
    score_chain = score_prompt | llm | StrOutputParser()
    scores_only = score_chain.invoke({"feedback_text": feedback_text})

    with open("data/interview_scores.txt", "w", encoding="utf-8") as f:
        f.write(scores_only)

    return {"status": "success", "message": "interview_scores.txt 파일로 저장됨"}