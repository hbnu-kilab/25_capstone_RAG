import logging
import uvicorn
import os
import openai

from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from openai import OpenAI
from config import load_api_key
from doc_retrieval.evaluation.evaluate_retrieval import search_documents
from doc_retrieval.dpr.model import Pooler
from doc_retrieval.database.vector_database import VectorDatabase
from transformers import AutoTokenizer, AutoModel


app = FastAPI(title="RAG API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key = load_api_key())

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발환경에서는 모든 origin 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RAGRequest(BaseModel):
    query: str
    is_consultant_mode: Optional[bool] = Field(default=False, alias="isConsultantMode")

    model_config = {"populate_by_name": True}

class RAGResponse(BaseModel):
    answer: str


# 일반 모드와 입시컨설턴트 모드 경로 설정
GENERAL_MODEL_PATH = "/app/doc_retrieval/model/question_encoder"
GENERAL_FAISS_PATH = "/app/doc_retrieval/data/faiss/faiss_pickle.pkl"
GENERAL_CONTEXT_PATH = "/app/doc_retrieval/data/faiss/context_pickle.pkl"

# 입시컨설턴트 모드 경로 (다른 데이터셋 사용)
CONSULTANT_MODEL_PATH = "/app/doc_retrieval/model/question_encoder"
CONSULTANT_FAISS_PATH = "/app/doc_retrieval/data/faiss/faiss_pickle.pkl"
CONSULTANT_CONTEXT_PATH = "/app/doc_retrieval/data/faiss/context_pickle.pkl"

def load_models_and_data(is_consultant_mode):
    """
    모드에 따라 모델과 데이터를 로드하는 함수
    
    Args:
        is_consultant_mode: 입시컨설턴트 모드 여부
        
    Returns:
        tuple: (tokenizer, q_encoder, pooler, faiss_db, faiss_index, text_index)
    """
    try:
        if is_consultant_mode:
            logger.info("입시컨설턴트 모드로 모델 로드 시도")
            model_path = CONSULTANT_MODEL_PATH
            faiss_path = CONSULTANT_FAISS_PATH
            context_path = CONSULTANT_CONTEXT_PATH
        else:
            logger.info("일반 모드로 모델 로드")
            model_path = GENERAL_MODEL_PATH
            faiss_path = GENERAL_FAISS_PATH
            context_path = GENERAL_CONTEXT_PATH
        
        # 파일 존재 확인
        if not os.path.exists(model_path):
            if is_consultant_mode:
                logger.warning(f"입시컨설턴트 모델이 없습니다: {model_path}. 일반 모드로 폴백")
                return load_models_and_data(is_consultant_mode=False)
            else:
                raise FileNotFoundError(f"모델 경로가 존재하지 않습니다: {model_path}")
        
        if not os.path.exists(faiss_path) or not os.path.exists(context_path):
            if is_consultant_mode:
                logger.warning(f"입시컨설턴트 데이터가 없습니다. 일반 모드로 폴백")
                return load_models_and_data(is_consultant_mode=False)
            else:
                raise FileNotFoundError(f"데이터 파일이 존재하지 않습니다: {faiss_path} 또는 {context_path}")
        
        # 모델 로드
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        q_encoder = AutoModel.from_pretrained(model_path)
        pooler = Pooler("cls")
        
        # 데이터베이스 로드
        faiss_db = VectorDatabase(
            faiss_pickle=faiss_path,
            context_pickle=context_path
        )
        
        faiss_index = faiss_db.faiss_index
        text_index = faiss_db.text_index
        
        logger.info(f"모델 및 데이터 로드 성공 - 경로: {model_path}")
        return tokenizer, q_encoder, pooler, faiss_db, faiss_index, text_index
        
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        if is_consultant_mode:
            logger.info("입시컨설턴트 모드 실패, 일반 모드로 폴백 시도")
            return load_models_and_data(is_consultant_mode=False)
        else:
            raise e
        
        
@app.post("/rag", response_model=RAGResponse)
def rag_generate(request: RAGRequest):
    """
    RAG 기반 응답 생성 엔드포인트
    """
    query = request.query
    is_consultant_mode = request.is_consultant_mode
    
    logger.info(f"RAG 요청 - 쿼리: {query[:50]}..., 입시컨설턴트 모드: {is_consultant_mode}")
    
    if not client:
        logger.error("OpenAI 클라이언트가 초기화되지 않음")
        raise HTTPException(status_code=500, detail="AI 서비스를 사용할 수 없습니다")
    
    try:
        # 모드에 따라 모델과 데이터 로드
        tokenizer, q_encoder, pooler, faiss_db, faiss_index, text_index = load_models_and_data(is_consultant_mode)
        
        # 1. 문서 검색
        logger.info("문서 검색 시작")
        doc_ids = search_documents(
            query, q_encoder, tokenizer, pooler,
            faiss_index, text_index,
            device='cpu'
        )
        
        # 2. 검색된 문서 ID를 실제 텍스트로 변환
        docs = []
        for doc_id in doc_ids[:5]:  # 상위 5개
            try:
                idx_position = text_index.index(doc_id)
                actual_text = faiss_db.text[idx_position]
                docs.append(actual_text)
                logger.debug(f"매칭 성공: {doc_id} -> 인덱스 {idx_position}")
            except (ValueError, IndexError) as e:
                logger.warning(f"문서 ID {doc_id}에 대한 텍스트를 찾을 수 없음: {e}")
                docs.append(f"텍스트 없음: {doc_id}")
        
        # 3. 컨텍스트 생성
        context = "\n\n--- 다음 문서 ---\n\n".join(docs)
        logger.info(f"최종 검색된 문서 개수: {len(docs)}")
        
        # 4. 모드에 따른 프롬프트 생성
        if is_consultant_mode:
            system_message = """당신은 전문적인 입시컨설턴트입니다. 
대학 입시, 학과 선택, 진로 상담에 대해 정확하고 도움이 되는 조언을 제공합니다."""
            
            prompt = f"""다음은 입시 관련 질문입니다: {query}

이 질문에 답하기 위해 다음 입시 관련 문서들을 참고하세요:
{context}

문서 기반으로 최대한 정확하게 답변하세요.
만약에 문서와 질문이 연관성이 없다고 판단이 되면 모른다고 답변하세요."""

        else:
            system_message = "당신은 전문적인 질문 응답 어시스턴트입니다."
            
            prompt = f"""다음은 사용자의 질문입니다: {query}

이 질문에 답하기 위해 다음 문서들을 참고하세요:
{context}

문서 기반으로 최대한 정확하게 답변하세요.
만약에 문서와 질문이 연관성이 없다고 판단이 되면 모른다고 답변하세요."""
        
        # 5. OpenAI API 호출
        logger.info("OpenAI API 호출 시작")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        response_content = completion.choices[0].message.content
        logger.info(f"RAG 응답 생성 완료 - 모드: {'입시컨설턴트' if is_consultant_mode else '일반'}")
        
        return RAGResponse(answer=response_content)
    
    except FileNotFoundError as e:
        logger.error(f"모델 또는 데이터 파일을 찾을 수 없음: {e}")
        raise HTTPException(status_code=500, detail=f"필요한 파일을 찾을 수 없습니다: {str(e)}")
    
    except Exception as e:
        logger.error(f"RAG 처리 중 오류 발생: {e}")
        import traceback
        logger.error(f"스택 트레이스: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"RAG 처리 오류 발생: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)