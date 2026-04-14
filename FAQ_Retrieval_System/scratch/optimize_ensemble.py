import json
import os
import sys
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
sys.path.append(os.getcwd())
from config import DATASET_FILE, GROUND_TRUTH_FILE, FINETUNED_SBERT_PATH
from utils.dataset import parse_dataset
from utils.preprocessor import call_c_preprocessor
from config import PREPROCESSOR_EXE

def calculate_overlap_score(list1, list2):
    if not list1 or not list2: return 0.0
    set1, set2 = set(list1), set(list2)
    intersection = set1.intersection(set2)
    return len(intersection) / max(len(set1), len(set2))

def optimize_weights():
    print("Loading FAQs...")
    faqs = parse_dataset(DATASET_FILE)
    print("Loading Test Cases...")
    with open(GROUND_TRUTH_FILE, "r") as f:
        test_cases = json.load(f)
    
    print("Pre-calculating features (this takes a moment)...")
    # Pre-process FAQ features
    for faq in faqs:
        analysis = call_c_preprocessor(PREPROCESSOR_EXE, faq["question"])
        faq["pos_f"] = analysis["pos"]
        faq["ner_f"] = analysis["ner"]
    
    # Load Fine-tuned SBERT
    model = SentenceTransformer(FINETUNED_SBERT_PATH if os.path.exists(FINETUNED_SBERT_PATH) else 'all-mpnet-base-v2')
    faq_embeddings = model.encode([f["question"] for f in faqs], show_progress_bar=True)
    
    # Pre-calculate test query features
    test_features = []
    for case in tqdm(test_cases, desc="Parsing test queries"):
        q = case['query']
        analysis = call_c_preprocessor(PREPROCESSOR_EXE, q)
        q_emb = model.encode([q])[0]
        test_features.append({
            'emb': q_emb,
            'pos': analysis['pos'],
            'ner': analysis['ner'],
            'target': case['target']
        })
        
    def evaluate(a, b, g):
        hits = 0
        rr = 0.0
        for feat in test_features:
            q_emb = feat['emb']
            q_pos = feat['pos']
            q_ner = feat['ner']
            target = feat['target']
            
            sims = cosine_similarity([q_emb], faq_embeddings).flatten()
            
            final_scores = []
            for i, faq in enumerate(faqs):
                pos_s = calculate_overlap_score(q_pos, faq["pos_f"])
                ner_s = calculate_overlap_score(q_ner, faq["ner_f"])
                s = (a * sims[i]) + (b * pos_s) + (g * ner_s)
                final_scores.append((i, s))
            
            final_scores.sort(key=lambda x: x[1], reverse=True)
            ranked_indices = [x[0] for x in final_scores]
            
            try:
                target_idx = next(i for i, f in enumerate(faqs) if f['question'] == target)
            except StopIteration:
                print(f"Warning: Target '{target}' not found in FAQs. Skipping.")
                continue
            
            rank = ranked_indices.index(target_idx) + 1
            if rank == 1: hits += 1
            rr += (1.0 / rank)
            
        return (hits / len(test_cases)) * 100, rr / len(test_cases)

    # Grid Search
    best_p1 = 0
    best_weights = (0, 0, 0)
    
    print("\nGrid searching weights (ALPHA, BETA, GAMMA)...")
    # We maintain a+b+g = 1.0
    for a in np.linspace(0.5, 0.9, 5):
        for b in np.linspace(0.0, 1.0 - a, 5):
            g = 1.0 - a - b
            p1, mrr = evaluate(a, b, g)
            print(f"A={a:.2f}, B={b:.2f}, G={g:.2f} -> P@1: {p1:.2f}%")
            if p1 > best_p1:
                best_p1 = p1
                best_weights = (a, b, g)
                
    print(f"\nBEST WEIGHTS: ALPHA={best_weights[0]:.2f}, BETA={best_weights[1]:.2f}, GAMMA={best_weights[2]:.2f}")
    print(f"Achieved P@1: {best_p1:.2f}%")

if __name__ == "__main__":
    optimize_weights()
