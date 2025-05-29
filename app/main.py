from fastapi import FastAPI
from app.question_gen import generate_questions
from app.feedback_gen import generate_feedback
from app.score_gen import generate_score

app = FastAPI(
    title="Interview Assistant API",
    description="포트폴리오 기반 질문 생성, 피드백, 점수화를 수행하는 API",
    version="1.0"
)

@app.get("/")
def read_root():
    return {"message": "인터뷰 지원 API입니다. /docs에서 테스트하세요."}

@app.post("/generate-questions")
def run_generate_questions():
    return generate_questions()

@app.post("/evaluate-feedback")
def run_generate_feedback():
    return generate_feedback()

@app.post("/score-feedback")
def run_generate_score():
    return generate_score()