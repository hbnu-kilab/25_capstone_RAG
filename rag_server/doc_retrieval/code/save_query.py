import json
import re
import openai
import requests

# 🔧 설정
load_dotenv(dotenv_path='rag_server/apikey.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

batch_id = "batch_id"  # 👉 배치 ID 입력
final_output_path = "rag_server/data/queries.jsonl"

# ✅ 질문 전처리 함수
def clean_question_text(q):
    return re.sub(r"^[-–—\s]*\d+[\.\)]*\s*|^[-–—•]\s*", "", q).strip()

# ✅ 질문 분할 및 저장 함수
def split_questions_and_write(data_lines, output_path):
    output = []
    id_counter = 1

    for line in data_lines:
        item = json.loads(line)
        source_id = item.get("_id", "")
        questions = item.get("text", "").strip().split("\n")
        for q in questions:
            cleaned_q = clean_question_text(q)
            if not cleaned_q:
                continue
            entry = {
                "_id": f"query-{id_counter:05d}",
                "text": cleaned_q,
                "metadata": {"source": source_id}
            }
            output.append(entry)
            id_counter += 1

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in output:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

# ✅ 전체 실행 흐름
def main():
    print(f"🔍 Checking batch status for ID: {batch_id}")
    batch = openai.Batch.retrieve(batch_id)

    if batch.status != "completed":
        print(f"⏳ Batch not ready. Current status: {batch.status}")
        return

    print("✅ Batch completed.")
    output_file_id = batch.output_file_id

    output_file = openai.File.retrieve(output_file_id)
    download_url = output_file.download_url

    print(f"📥 Downloading from: {download_url}")
    response = requests.get(download_url)
    data_lines = response.text.strip().splitlines()

    split_questions_and_write(data_lines, final_output_path)
    print(f"✅ Final cleaned output saved to '{final_output_path}'")

if __name__ == "__main__":
    main()