from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.logger import logger
from pydantic import BaseModel
from typing import Optional, List
import logging
import uvicorn

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KILAB Chatbot API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js 프론트엔드 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatMessage(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    role: str
    content: str

@app.get("/")
async def root():
    return JSONResponse({"message": "KILAB Chatbot API가 실행 중입니다. /docs에서 API 문서를 확인하세요."})

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    logger.info(f"API 호출: {message}")
    try:
        query = message.messages[-1].content
        
        if query == "안녕":
            answer = "안녕하세요! 무엇을 도와드릴까요?"
        else:
            answer = f"죄송합니다. {query}에 대해 답변을 찾을 수 없습니다."
        
        response = ChatResponse(
            role="assistant",
            content=f"{answer}"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
