import openai
import os
from dotenv import load_dotenv
from openai import OpenAI

import json

def prepare_batch_input_file(corpus_path, batch_input_path):
    with open(corpus_path, 'r', encoding='utf-8') as corpus_file, \
         open(batch_input_path, 'w', encoding='utf-8') as batch_file:

        for line in corpus_file:
            doc = json.loads(line)
            doc_id = doc.get("_id", "")
            text = doc.get("text", "")

            prompt = f"""다음의 <order>에 따라 문서 내용을 기반으로 해당 문서의 정보를 찾는 질문을 생성해주세요.
<order>
반드시 문서에 명시되어있는 지식에서만 질문을 설계합니다.
2. 문서의 질문에는 구체적으로 원하는 답에 대한 정보가 충분히 포함되어야합니다.
3. 질문은 문서에서 나올 수 있는 모든 정보에 대한 질문을 모두 작성합니다.
4. 절대로 문서 내용에서 설명 불가능한 질문은 생성하지 않습니다.
5. 최종 답변은 질문만 생성합니다.

[문서 내용]
{text}
"""

            messages = [
                {"role": "system", "content": "당신은 국립한밭대학교에 재학중인 학생입니다. 학교에 대해 궁금한 점을 질문으로 생성합니다."},
                {"role": "user", "content": prompt}
            ]

            entry = {
                "custom_id": f"{doc_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",  # 또는 "gpt-4" 등 사용하고자 하는 모델
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 100
                }
            }

            batch_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

# 사용 예시
prepare_batch_input_file("rag_server/data/corpus.jsonl", "rag_server/data/query_batch.jsonl")
