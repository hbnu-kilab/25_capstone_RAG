import openai
import os
from dotenv import load_dotenv
from openai import OpenAI

# API 키 설정
load_dotenv(dotenv_path='rag_server/apikey.env')
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# 경로 설정
INPUT_JSONL = "rag_server/data/query_batch.jsonl"

# 1. 파일 업로드
with open(INPUT_JSONL, "rb") as f:
    uploaded_file = client.files.create(
        file=f,
        purpose="batch"
    )

# 2. 배치 제출
batch = client.batches.create(
    input_file_id=uploaded_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

print(f"✅ 배치 제출 완료! batch id: {batch.id}")
print(f"🕐 상태: {batch.status}")
print(f"📌 결과 URL (완료 후): {batch.output_file_id}")