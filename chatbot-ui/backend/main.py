import os
import json
import logging
import uvicorn
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from config import load_api_key
from llm_cache import LLMCache

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

# ChromaDB 클라이언트 초기화
chroma_client = chromadb.Client()

# API 키 로드
def load_api_key():
    try:
        with open('api_key.json', 'r') as f:
            api_keys = json.load(f)
            return api_keys['openai_api_key']
    except Exception as e:
        logger.error(f"API 키 로드 실패: {e}")
        raise

# OpenAI Embedding 함수 초기화
openai_ef = OpenAIEmbeddingFunction(
    api_key=load_api_key(),
    model_name="text-embedding-ada-002"
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
        
        start_time = time.time()
        answer = app.llm_cache.generate(query)
        
        print(f'질문: {query}')
        print("소요 시간: {:.2f}s".format(time.time() - start_time))
        print(f'답변: {answer}\n')
        
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
