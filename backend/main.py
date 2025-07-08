import os
import json
import logging
import uvicorn
import time

# ChromaDB 텔레메트리 비활성화
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List

import openai  # OpenAI 0.28 방식
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from config import load_api_key
from llm_cache import LLMCache

# # 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KILAB Chatbot API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://ki-chat:3000",  # Docker 내부 통신
        "http://localhost:3000",  # 로컬 개발
        "*"  # 개발환경에서 모든 origin 허용
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    api_key = load_api_key()
    openai.api_key = api_key
    logger.info("OpenAI API 키 설정 성공 (v0.28)")
except Exception as e:
    logger.error(f"OpenAI API 키 설정 실패: {e}")
    openai.api_key = None
    
    
# ChromaDB 클라이언트 초기화
chroma_client = chromadb.Client()

# OpenAI Embedding 함수 초기화
openai_ef = OpenAIEmbeddingFunction(
    api_key=load_api_key(),
    model_name="text-embedding-3-small"
)

# Semantic Cache 초기화
semantic_cache = chroma_client.get_or_create_collection(
    name="semantic_cache",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"}
)

# LLM Cache 초기화
app.llm_cache = LLMCache(semantic_cache)

class Message(BaseModel):
    role: str
    content: str
    is_consultant_mode: Optional[bool] = Field(default=False, alias="consultantMode")
    
    model_config = {"populate_by_name": True}

class ChatRequest(BaseModel):
    messages: List[Message]
    is_consultant_mode: Optional[bool] = Field(default=False, alias="isConsultantMode")

    model_config = {"populate_by_name": True}

class ChatResponse(BaseModel):
    role: str
    content: str

@app.get("/")
async def root():
    return JSONResponse({"message": "KILAB Chatbot API가 실행 중입니다. /docs에서 API 문서를 확인하세요."})

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f"API 호출: {request}")
    logger.info(f"받은 메시지 수: {len(request.messages)}")
    
    try:
        user_messages = [msg for msg in request.messages if msg.role == "user" and msg.content.strip()]
        
        if not user_messages:
            raise HTTPException(status_code=400, detail="유효한 사용자 메시지가 없습니다")
        
        # 마지막 사용자 메시지 가져오기
        last_message = user_messages[-1]
        query = last_message.content
        
        # 입시컨설턴트 모드 확인 (요청 레벨과 메시지 레벨 모두 확인)
        is_consultant_mode = request.is_consultant_mode or last_message.is_consultant_mode
        
        logger.info(f"질문: {query}")
        logger.info(f"입시컨설턴트 모드: {is_consultant_mode}")
        
        start_time = time.time()
        answer = app.llm_cache.generate(query, is_consultant_mode)
        
        elapsed_time = time.time() - start_time
        logger.info(f"소요 시간: {elapsed_time:.2f}s")
        logger.info(f"답변: {answer[:100]}...")
        
        response = ChatResponse(
            role="assistant",
            content=answer
        )
        
        return response
    
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)