from dotenv import load_dotenv
load_dotenv()


from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

def load_pdf_with_pypdf2(file_path: str) -> list[Document]:
    reader = PdfReader(file_path)
    return [Document(page_content=page.extract_text() or "") for page in reader.pages]
def generate_questions():
    docs = load_pdf_with_pypdf2("data/data.pdf")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert IT recruiter. Summarize the main projects and technical stacks from the following portfolio text. Respond in bullet points. Only return the summary, no explanations."),
        ("human", "{input}")
    ])
    summary_chain = summary_prompt | llm | StrOutputParser()
    summaries = [summary_chain.invoke({"input": chunk.page_content}) for chunk in chunks]
    full_summary = "\n".join(summaries)

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

    with open("data/interview_questions.txt", "w", encoding="utf-8") as f:
        f.write(questions)

    return {"status": "success", "message": "interview_questions.txt 파일로 저장 완료됨"}