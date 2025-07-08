import logging
import openai
import requests
from fastapi import HTTPException


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMCache:
    def __init__(self, semantic_cache):
        self.cache = {}
        self.semantic_cache = semantic_cache

    def generate(self, query, is_consultant_mode):
        if query in self.cache:
            return self.cache[query]
        
        similar_docs = self.semantic_cache.query(
            query_texts=[query],
            n_results=1
        )
        
        # 유사한 질문이 있고 유사도가 임계값(0.*) 이하인 경우
        if (len(similar_docs['distances'][0]) > 0 and 
            similar_docs['distances'][0][0] < 0.05):
            return similar_docs['metadatas'][0][0]['response']
        
        response = self.response_to_rag(query, is_consultant_mode)
        response_str = self._format_response(response)
        
        # 캐시에 저장
        self.cache[query] = response_str
        self.semantic_cache.add(
            documents=[query],
            metadatas=[{"response": response_str}],
            ids=[query]
        )
        
        return response_str
    
    def response_to_rag(self, query, is_consultant_mode):
        try:
            
            response = requests.post(
                "http://rag:8001/rag",
                json={"query": query, "is_consultant_mode": is_consultant_mode},
                timeout=30
            )
            
            return response.json()
        
            if response.status_code != 200:
                raise Exception(f"RAG API 요청 실패: {response.status_code} {response.text}")
        
        except Exception as e:
            logger.error(f"오류 발생: {e}")
            raise HTTPException(status_code=500, detail=str(e), response=None, body=None)
    
    
    def _format_response(self, response):
        if isinstance(response, dict) and 'answer' in response:
            return response['answer']
        if isinstance(response, str):
            return response
        return str(response)