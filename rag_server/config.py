import json

def load_api_key():
    try:
        with open('api_key.json', 'r') as f:
            api_keys = json.load(f)
            return api_keys['openai_api_key']
    except Exception as e:
        raise Exception(f"API 키 로드 실패: {str(e)}")
