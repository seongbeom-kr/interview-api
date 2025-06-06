from fastapi import FastAPI
import os
from app.question_gen import generate_questions
from app.feedback_gen import generate_feedback
from app.score_gen import generate_score
from app.upload import router as upload_router

app = FastAPI(
    title="Interview Assistant API",
    description="포트폴리오 기반 질문 생성, 피드백, 점수화를 수행하는 API",
    version="1.0"
)

app.include_router(upload_router)

@app.get("/")
def read_root():
    return {"message": "인터뷰 지원 API입니다. /docs에서 테스트하세요."}

@app.post("/generate-questions")
def run_generate_questions():
    return generate_questions("data/data.pdf")

@app.get("/get-questions-text")
def get_questions_text():
    file_path = "data/interview_questions.txt"
    if not os.path.exists(file_path):
        return {"status": "error", "message": "질문 파일이 아직 생성되지 않았습니다."}
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return {
        "status": "success",
        "questions": content
    }

@app.post("/evaluate-feedback")
def run_generate_feedback():
    return generate_feedback("data/interview_qa.txt")

@app.get("/get-feedback-text")
def get_feedback_text():
    file_path = "data/interview_feedback.txt"
    if not os.path.exists(file_path):
        return {"status": "error", "message": "피드백 파일이 아직 생성되지 않았습니다."}

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return {
        "status": "success",
        "feedback": content
    }

@app.post("/score-feedback")
def run_generate_score():
    return generate_score("data/interview_feedback.txt")

@app.get("/get-score-text")
def get_score_text():
    file_path = "data/interview_scores.txt"
    if not os.path.exists(file_path):
        return {"status": "error", "message": "점수 파일이 아직 생성되지 않았습니다."}

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return {
        "status": "success",
        "score": content
    }