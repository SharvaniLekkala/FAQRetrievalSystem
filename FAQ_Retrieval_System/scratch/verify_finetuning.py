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
from config import DATASET_FILE, GROUND_TRUTH_FILE, FINETUNED_SBERT_PATH
from utils.dataset import parse_dataset

def evaluate_model(model_name_or_path, faqs, test_cases, label):
    print(f"Evaluating {label}...")
    model = SentenceTransformer(model_name_or_path)
    
    faq_questions = [f["question"] for f in faqs]
    faq_embeddings = model.encode(faq_questions, show_progress_bar=True)
    
    hits = 0
    rr_sum = 0.0
    
    for case in tqdm(test_cases, desc=f"Eval {label}"):
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
    
    results = []
    
    # Baseline: all-MiniLM-L6-v2 (Initial)
    p1_mini, mrr_mini = evaluate_model('all-MiniLM-L6-v2', faqs, test_cases, "Initial MiniLM")
    results.append(("MiniLM (Base)", p1_mini, mrr_mini))
    
    # Strong Baseline: all-mpnet-base-v2 (Pre-trained)
    p1_mpnet, mrr_mpnet = evaluate_model('all-mpnet-base-v2', faqs, test_cases, "Pre-trained MPNet")
    results.append(("MPNet (Base)", p1_mpnet, mrr_mpnet))
    
    # Final: Fine-tuned model
    if os.path.exists(FINETUNED_SBERT_PATH):
        p1_ft, mrr_ft = evaluate_model(FINETUNED_SBERT_PATH, faqs, test_cases, "Fine-tuned MPNet")
        results.append(("MPNet (Fine-tuned)", p1_ft, mrr_ft))
    
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    print(f"{'MODEL':<20} | {'P@1 (%)':<15} | {'MRR':<10}")
    print("-" * 60)
    for model, p1, mrr in results:
        print(f"{model:<20} | {p1:>8.2f}%       | {mrr:>8.4f}")
    print("="*60)
