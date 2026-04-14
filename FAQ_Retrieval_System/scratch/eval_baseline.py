import json
import os
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
import sys
sys.path.append(os.getcwd())
from config import DATASET_FILE, GROUND_TRUTH_FILE
from utils.dataset import parse_dataset

def evaluate_model(model_name, faqs, test_cases):
    print(f"Evaluating {model_name}...")
    model = SentenceTransformer(model_name)
    
    faq_questions = [f["question"] for f in faqs]
    faq_embeddings = model.encode(faq_questions, show_progress_bar=True)
    
    hits = 0
    rr_sum = 0.0
    
    for case in tqdm(test_cases, desc="Running Eval"):
        query = case['query']
        target = case['target']
        
        query_emb = model.encode([query])
        sims = cosine_similarity(query_emb, faq_embeddings).flatten()
        
        ranked_indices = np.argsort(sims)[::-1]
        
        try:
            target_idx = next(i for i, f in enumerate(faqs) if f['question'] == target)
            rank = np.where(ranked_indices == target_idx)[0][0] + 1
            if rank == 1: hits += 1
            rr_sum += (1.0 / rank)
        except StopIteration:
            continue
            
    p1 = (hits / len(test_cases)) * 100
    mrr = (rr_sum / len(test_cases))
    return p1, mrr

if __name__ == "__main__":
    faqs = parse_dataset(DATASET_FILE)
    with open(GROUND_TRUTH_FILE, "r") as f:
        test_cases = json.load(f)
    
    # Baseline 1: current model
    p1_mini, mrr_mini = evaluate_model('all-MiniLM-L6-v2', faqs, test_cases)
    
    # Baseline 2: mpnet model (pre-trained)
    p1_mpnet, mrr_mpnet = evaluate_model('all-mpnet-base-v2', faqs, test_cases)
    
    print("\n" + "="*40)
    print("BASELINE PERFORMANCE")
    print("="*40)
    print(f"all-MiniLM-L6-v2: P@1: {p1_mini:.2f}%, MRR: {mrr_mini:.4f}")
    print(f"all-mpnet-base-v2: P@1: {p1_mpnet:.2f}%, MRR: {mrr_mpnet:.4f}")
    print("="*40)
