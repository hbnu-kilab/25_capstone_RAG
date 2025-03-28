import openai
from config import load_api_key

client = openai.OpenAI(api_key=load_api_key())

class LLMCache:
    def __init__(self, semantic_cache):
        self.cache = {}
        self.semantic_cache = semantic_cache

    def generate(self, query):
        
        if query in self.cache:
            return self.cache[query]
        
        similar_docs = self.semantic_cache.query(
            query_texts=[query],
            n_results=1
        )
        
        # 유사한 질문이 있고 유사도가 임계값(0.*) 이하인 경우
        if (len(similar_docs['distances'][0]) > 0 and 
            similar_docs['distances'][0][0] < 0.1):
            return similar_docs['metadatas'][0][0]['response']
        
        response = self.response_to_rag(query)
        response_str = self._format_response(response)
        
        # 캐시에 저장
        self.cache[query] = response_str
        self.semantic_cache.add(
            documents=[query],
            metadatas=[{"response": response_str}],
            ids=[query]
        )
        
        return response_str
    
    def response_to_rag(self, query):
        # 추후 구현할 것
        # 임시로 응답 반환 테스트
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 웹 검색을 할 수 있는 도구를 사용할 수 있습니다.",
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )
        response = completion.choices[0].message.content
        return response
    
    def _format_response(self, response):
        if isinstance(response, str):
            return response
        return str(response)
