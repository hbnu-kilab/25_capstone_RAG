import sys
import faiss
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import time  # 시간 측정용

sys.path.append('../')
from doc_retrieval.dpr import Pooler
from doc_retrieval.dpr import BiEncoderDataset
from doc_retrieval.database.vector_database import VectorDatabase
from doc_retrieval.utils.bm25 import BM25Reranker
from doc_retrieval.utils.utils import get_topk_accuracy

LOGGER = logging.getLogger()

def argument_parser():
    parser = argparse.ArgumentParser(description='get topk-accuracy of retrieval model')
    
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--valid_data', type=str, required=True)
    parser.add_argument('--faiss_path', type=str, required=True)
    parser.add_argument('--bm25_path', type=str, required=False)
    parser.add_argument('--faiss_weight', default=1, type=float)
    parser.add_argument('--bm25_weight', default=0.5, type=float)
    parser.add_argument('--search_k', default=2000, type=int)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--pooler', default='cls', type=str)
    parser.add_argument('--padding', action="store_false", default=True)
    parser.add_argument('--truncation', action="store_false", default=True)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    
    return parser.parse_args()

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def search_evaluation(q_encoder, tokenizer, test_dataset, faiss_index, text_index,
                      search_k=2000, bm25_model=None, faiss_weight=1, bm25_weight=0.5, max_length=512, 
                      pooler=None, padding=True, truncation=True, batch_size=32, device='cuda'):
    
    question = test_dataset.question
    answer_idx = test_dataset.answer_idx

    q_encoder = q_encoder.to(device)
    q_encoder.eval()
    
    question_embed = []
    inference_times = []  # 시간 측정용

    for start_index in tqdm(range(0, len(question), batch_size)):
        batch_question = question[start_index : start_index + batch_size]

        q_batch = tokenizer(batch_question,
                            padding=padding,
                            max_length=max_length,
                            truncation=truncation,
                            return_tensors='pt')

        # 시간 측정 시작
        start_time = time.perf_counter()

        with torch.no_grad():
            q_output = q_encoder(input_ids=q_batch['input_ids'].to(device),
                                 attention_mask=q_batch['attention_mask'].to(device),
                                 token_type_ids=q_batch['token_type_ids'].to(device))
        
        if pooler:
            pooler_output = pooler(q_batch['attention_mask'], q_output).cpu()
        else:
            pooler_output = q_output.last_hidden_state[:, 0, :].cpu()

        # 시간 측정 종료
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000  # ms
        inference_times.extend([elapsed_time] * len(batch_question))

        question_embed.append(pooler_output)
     
    question_embed = np.vstack(question_embed)

    print('>>> Searching documents using faiss index.')
    D, I = faiss_index.search(question_embed, search_k)

    if bm25_model:
        print('>>> Reranking candidates with BM25 scores.')
        bm25_scores = bm25_model.get_bm25_rerank_scores(question, I)
        total_scores = faiss_weight * D + bm25_weight * bm25_scores

        for idx in range(total_scores.shape[0]):
            sorted_idx = np.argsort(total_scores[idx])[::-1]
            I[idx] = I[idx][sorted_idx]

    scores = get_topk_accuracy(I, answer_idx, text_index)

    total_time = sum(inference_times)
    avg_time = total_time / len(question)

    scores['total_inference_time_ms'] = total_time
    scores['avg_inference_time_per_query_ms'] = avg_time

    return scores

def main(args):
    init_logging()
    
    LOGGER.info('*** Top-k Retrieval Accuracy ***')
    
    q_encoder = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pooler = Pooler(args.pooler)
    
    test_dataset = BiEncoderDataset.load_valid_dataset(args.valid_data)

    faiss_vector = VectorDatabase(args.faiss_path)
    faiss_index = faiss_vector.faiss_index
    text_index = faiss_vector.text_index

    bm25_model = BM25Reranker(bm25_pickle=args.bm25_path) if args.bm25_path else None

    scores = search_evaluation(q_encoder, tokenizer, test_dataset, faiss_index, text_index, 
                               search_k=args.search_k, bm25_model=bm25_model,
                               faiss_weight=args.faiss_weight, bm25_weight=args.bm25_weight,
                               max_length=args.max_length, pooler=pooler,
                               padding=args.padding, truncation=args.truncation,
                               batch_size=args.batch_size, device=args.device)

    print()
    print('=== Top-k Accuracy ===')
    print(f"Top1 Acc: {scores['top1_accuracy']*100:.2f} (%)")
    print(f"Top2 Acc: {scores['top2_accuracy']*100:.2f} (%)")
    print(f"Top3 Acc: {scores['top3_accuracy']*100:.2f} (%)")
    print(f"Top5 Acc: {scores['top5_accuracy']*100:.2f} (%)")
    print(f"Top10 Acc: {scores['top10_accuracy']*100:.2f} (%)")
    print(f"Top20 Acc: {scores['top20_accuracy']*100:.2f} (%)")
    print(f"Total Inference Time: {scores['total_inference_time_ms']:.2f} (ms)")
    print(f"Avg Inference Time: {scores['avg_inference_time_per_query_ms']:.2f} (ms)")
    print('======================')


# search_engine.py
def search_documents(query: str, 
                     q_encoder, tokenizer, pooler, faiss_index, text_index, 
                     bm25_model=None, 
                     max_length=512, 
                     faiss_weight=1.0, bm25_weight=0.5,
                     search_k=10, device='cuda'):
    q_encoder = q_encoder.to(device)
    q_encoder.eval()
    
    q_batch = tokenizer([query],
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors='pt')
    
    with torch.no_grad():
        output = q_encoder(input_ids=q_batch['input_ids'].to(device),
                           attention_mask=q_batch['attention_mask'].to(device),
                           token_type_ids=q_batch['token_type_ids'].to(device))

    if pooler:
        embedding = pooler(q_batch['attention_mask'], output).cpu().numpy()
    else:
        embedding = output.last_hidden_state[:, 0, :].cpu().numpy()

    D, I = faiss_index.search(embedding, search_k)

    if bm25_model:
        bm25_scores = bm25_model.get_bm25_rerank_scores([query], I)
        total_scores = faiss_weight * D + bm25_weight * bm25_scores
        sorted_idx = np.argsort(total_scores[0])[::-1]
        I = I[:, sorted_idx]

    retrieved_texts = [text_index[idx] for idx in I[0]]
    return retrieved_texts

if __name__ == '__main__':
    args = argument_parser()
    main(args)