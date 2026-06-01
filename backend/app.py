import os
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pypdf import PdfReader

from rag_utils import (
    answer_with_rag,
    answer_with_zero_shot,
    load_index,
    match_resume_to_companies,
)

# ── PATH FIX ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")
# ──────────────────────────────────────────────────────────

faiss_index, docs = load_index()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


class ChatRequest(BaseModel):
    question: str
    mode: str = "rag"


class SourceItem(BaseModel):
    year: Optional[str] = None
    department: Optional[str] = None
    company: Optional[str] = None
    score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    if req.mode == "zero":
        result = answer_with_zero_shot(req.question)
    else:
        result = answer_with_rag(req.question, faiss_index, docs)

    sources = [
        SourceItem(
            year=s.get("year"),
            department=s.get("department"),
            company=s.get("company"),
            score=s.get("score")
        )
        for s in result.get("sources", [])
    ]
    print(result)

    return ChatResponse(
        answer=result["answer"],
        sources=sources
    )

@app.post("/api/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    try:
        # Read the uploaded PDF file
        pdf_reader = PdfReader(file.file)
        resume_text = ""
        
        # Extract text out of every page of the PDF
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                resume_text += text + " "
                
        if not resume_text.strip():
            return {"success": False, "message": "Could not extract text from this PDF. Is it scanned/an image?"}
            
        # Get matching companies using our function
        suggestions = match_resume_to_companies(resume_text, docs)
        
        return {
            "success": True,
            "suggestions": suggestions
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error processing PDF: {str(e)}"}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
