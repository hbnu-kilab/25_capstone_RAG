import os
import json
import logging
import uvicorn
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import openai
from config import load_api_key

app = FastAPI(title="RAG API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


openai.api_key = load_api_key()

class RAGRequest(BaseModel):
    query: str

class RAGResponse(BaseModel):
    answer: str

@app.post("/rag", response_model=RAGResponse)
def rag_generate(request: RAGRequest):
    query = request.query
    
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 웹 검색을 할 수 있는 도구를 사용할 수 있습니다."},
                {"role": "user", "content": query},
            ],
        )
        response = completion.choices[0].message.content
        return RAGResponse(answer=response)
    
    except Exception as e:
        logger.error(f"RAG 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="RAG 처리 오류 발생")

# FastAPI 실행용 코드
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)