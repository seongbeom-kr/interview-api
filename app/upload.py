from fastapi import APIRouter, UploadFile, File
import os
import shutil
from app.question_gen import generate_questions
from app.feedback_gen import generate_feedback
from app.score_gen import generate_score

router = APIRouter()
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "../data")

@router.post("/upload/pdf")
async def upload_pdf_and_generate(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_DIR, "data.pdf")
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    question_result = generate_questions(save_path)
    return {
        "status": "success",
        "message": f"{file.filename} 업로드 완료 및 질문 생성 성공",
        "question_status": question_result
    }

@router.post("/upload/qa")
async def upload_qa_and_feedback_score(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_DIR, "interview_qa.txt")
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    feedback_result = generate_feedback(save_path)
    score_result = generate_score("data/interview_feedback.txt")

    return {
        "status": "success",
        "message": f"{file.filename} 업로드 완료 및 피드백 + 점수 산출 완료",
        "feedback_status": feedback_result,
        "score_status": score_result
    }