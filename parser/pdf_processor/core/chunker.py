
from typing import List

class TextChunker:
    @staticmethod
    def chunk_text(text: str, chunk_size: int) -> List[str]:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
        for para in paragraphs:
            if len(para) <= chunk_size:
                chunks.append(para)
            else:
                words = para.split()
                current_chunk = []
                current_length = 0
                for word in words:
                    word_length = len(word) + 1
                    if current_length + word_length <= chunk_size:
                        current_chunk.append(word)
                        current_length += word_length
                    else:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [word]
                        current_length = word_length
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
        return chunks