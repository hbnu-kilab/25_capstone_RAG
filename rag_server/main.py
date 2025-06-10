import logging
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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

class RAGRequest(BaseModel):
    query: str

class RAGResponse(BaseModel):
    answer: str

MODEL_PATH = "/app/doc_retrieval/model/question_encoder"
FAISS_PATH = "/app/doc_retrieval/data/faiss/faiss_pickle.pkl"
CONTEXT_PATH = "/app/doc_retrieval/data/faiss/context_pickle.pkl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
q_encoder = AutoModel.from_pretrained(MODEL_PATH)
pooler = Pooler("cls")  # 또는 mean 등

faiss_db = VectorDatabase(
    faiss_pickle=FAISS_PATH,
    context_pickle=CONTEXT_PATH)

# bm25_model = BM25Reranker(bm25_pickle=BM25_PATH)
faiss_index = faiss_db.faiss_index
text_index = faiss_db.text_index

@app.post("/rag", response_model=RAGResponse)
def rag_generate(request: RAGRequest):
    query = request.query
    
    try:
        # 1. 검색 (ID들이 반환됨)
        doc_ids = search_documents(query, q_encoder, tokenizer, pooler,
                                 faiss_index, text_index,
                                 device='cpu')

        # 2. ID를 실제 텍스트로 올바르게 매칭
        docs = []
        for doc_id in doc_ids[:5]:
            try:
                # text_index에서 해당 ID의 위치(인덱스) 찾기
                idx_position = text_index.index(doc_id)
                # 그 위치의 실제 텍스트 가져오기
                actual_text = faiss_db.text[idx_position]
                docs.append(actual_text)
                logger.info(f"매칭 성공: {doc_id} -> 인덱스 {idx_position}")
            except (ValueError, IndexError) as e:
                logger.warning(f"문서 ID {doc_id}에 대한 텍스트를 찾을 수 없음: {e}")
                docs.append(f"텍스트 없음: {doc_id}")

        # 3. 프롬프트 생성
        context = "\n\n--- 다음 문서 ---\n\n".join(docs)
        logger.info(f"최종 검색된 문서 개수: {len(docs)}")
        
        prompt = f"""사용자의 질문에 대해 문서 기반으로 최대한 정확하게 답변하세요. \n\n 다음은 사용자의 질문입니다:\n{query}\n\n 이 질문에 답하기 위해 다음 문서들을 참고하세요:\n{context}\n\n """

        # 3. LLM 호출
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 전문적인 질문 응답 어시스턴트입니다."},
                {"role": "user", "content": prompt},
            ],
        )
        response = completion.choices[0].message.content
        return RAGResponse(answer=response)
    
    except Exception as e:
        logger.error(f"RAG 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="RAG 처리 오류 발생")
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)