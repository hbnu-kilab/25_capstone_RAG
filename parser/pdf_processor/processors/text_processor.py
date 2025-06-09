import fitz
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.logger import setup_logger

class TextProcessor:
    def __init__(self, overlap_threshold: float = 0.7):
        self.overlap_threshold = overlap_threshold
        self.logger = setup_logger(self.__class__.__name__)
    
    def process_text_blocks(self, page: fitz.Page, table_rects: List) -> List[Dict[str, Any]]:
        """테이블과 겹치지 않는 텍스트 블록을 처리합니다."""
        elements = []
        
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:  # 텍스트 블록
                block_bbox = fitz.Rect(block["bbox"])
                overlaps_with_table = False
                
                for table_rect in table_rects:
                    intersection = block_bbox & table_rect
                    if intersection:
                        overlap_ratio = intersection.get_area() / block_bbox.get_area()
                        if overlap_ratio > self.overlap_threshold:
                            overlaps_with_table = True
                            break
                
                if not overlaps_with_table:
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"]
                    if block_text.strip():
                        elements.append({
                            "type": "text",
                            "content": block_text,
                            "bbox": block["bbox"],
                            "y": block["bbox"][1]
                        })
        
        return elements