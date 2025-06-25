import datetime
import numpy as np

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed))) # Round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss

import time

def get_topk_accuracy(faiss_index, answer_idx, positive_idx): 
    top1_correct = 0
    top2_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    top20_correct = 0

    start_time = time.perf_counter()  # 더 정밀한 시간 측정 시작

    for idx, answer in enumerate(answer_idx):
        retrieved_idx = faiss_index[idx]
        retrieved_idx = [positive_idx[jdx] for jdx in retrieved_idx]

        if any(ridx in answer for ridx in retrieved_idx[:1]):
            top1_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:2]):
            top2_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:3]):
            top3_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:5]):
            top5_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:10]):
            top10_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:20]):
            top20_correct += 1

    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000  # 전체 소요 시간 (ms)
    avg_time_per_query_ms = total_time_ms / len(answer_idx)

    top1_accuracy = top1_correct / len(answer_idx)
    top2_accuracy = top2_correct / len(answer_idx)
    top3_accuracy = top3_correct / len(answer_idx)    
    top5_accuracy = top5_correct / len(answer_idx)
    top10_accuracy = top10_correct / len(answer_idx)
    top20_accuracy = top20_correct / len(answer_idx)

    return {
        'top1_accuracy': top1_accuracy,
        'top2_accuracy': top2_accuracy,
        'top3_accuracy': top3_accuracy,
        'top5_accuracy': top5_accuracy,
        'top10_accuracy': top10_accuracy,
        'top20_accuracy': top20_accuracy,
        'total_inference_time_ms': total_time_ms,
        'avg_inference_time_per_query_ms': avg_time_per_query_ms
    }

def get_topk_accuracy_cross(predictions, answer_idx):
    """Top-k 정확도 계산 함수"""
    top1_correct = 0
    top2_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    top20_correct = 0


    # 1 2 3 5 10 20
    for idx, answer in enumerate(answer_idx):
        retrieved_idx = predictions[idx]
        
        if any(ridx in answer for ridx in retrieved_idx[:1]):
            top1_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:2]):
            top2_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:3]):
            top3_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:5]):
            top5_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:10]):
            top10_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:20]):
            top20_correct += 1
    
    total = len(answer_idx)

    return {
        'top1_accuracy': top1_correct / total,
        'top2_accuracy': top2_correct / total,
        'top3_accuracy': top3_correct / total,
        'top5_accuracy': top5_correct / total,
        'top10_accuracy': top10_correct / total,
        'top20_accuracy': top20_correct / total,
    }