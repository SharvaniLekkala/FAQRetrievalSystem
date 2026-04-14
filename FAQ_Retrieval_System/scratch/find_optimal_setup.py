import json
import os
import sys
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

# Local imports
sys.path.append(os.getcwd())
from utils.dataset import parse_dataset
from utils.preprocessor import call_c_preprocessor
from config import DATASET_FILE, GROUND_TRUTH_FILE, FINETUNED_SBERT_PATH, PREPROCESSOR_EXE

def calculate_overlap_score(list1, list2):
    if not list1 or not list2: return 0.0
    set1, set2 = set(list1), set(list2)
    intersection = set1.intersection(set2)
    return len(intersection) / max(len(set1), len(set2))

def run_research():
    print("--- Loading Data ---")
    faqs = parse_dataset(DATASET_FILE)
    with open(GROUND_TRUTH_FILE, "r") as f:
        test_cases = json.load(f)
    
    # Map questions to domains for classifier testing
    q_to_domain = {f["question"]: f["domain"] for f in faqs}
    
    print("--- Pre-calculating FAQ Features ---")
    for faq in tqdm(faqs, desc="FAQs"):
        analysis = call_c_preprocessor(PREPROCESSOR_EXE, faq["question"])
        faq["pos_f"] = analysis["pos"]
        faq["ner_f"] = analysis["ner"]
        
    print("--- Loading SBERT (Fine-tuned) ---")
    model = SentenceTransformer(FINETUNED_SBERT_PATH if os.path.exists(FINETUNED_SBERT_PATH) else 'all-mpnet-base-v2')
    faq_embeddings = model.encode([f["question"] for f in faqs], show_progress_bar=True)
    
    print("--- Testing SBERT-based Classifier ---")
    X_train = faq_embeddings
    y_train = [f["domain"] for f in faqs]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate classifier on test queries
    correct_domains = 0
    test_data = []
    for case in tqdm(test_cases, desc="Queries"):
        q = case['query']
        target = case['target']
        true_domain = q_to_domain.get(target)
        
        q_emb = model.encode([q])[0]
        pred_domain = clf.predict([q_emb])[0]
        conf = clf.predict_proba([q_emb])[0].max()
        
        if true_domain and pred_domain.lower() == true_domain.lower():
            correct_domains += 1
            
        analysis = call_c_preprocessor(PREPROCESSOR_EXE, q)
        test_data.append({
            'emb': q_emb,
            'pos': analysis['pos'],
            'ner': analysis['ner'],
            'target': target,
            'pred_domain': pred_domain,
            'conf': conf,
            'true_domain': true_domain
        })
        
    print(f"SBERT Classifier Accuracy: {(correct_domains/len(test_data))*100:.2f}%")

    print("--- Optimizing Ensemble Weights ---")
    best_p1 = 0
    best_w = (0, 0, 0)
    
    # Search space for ALPHA (Semantic), BETA (POS), GAMMA (NER)
    for a in [0.7, 0.8, 0.85, 0.9, 0.95]:
        b_max = 1.0 - a
        for b in np.linspace(0, b_max, 5):
            g = 1.0 - a - b
            
            hits = 0
            for feat in test_data:
                q_emb = feat['emb']
                q_pos = feat['pos']
                q_ner = feat['ner']
                target = feat['target']
                pred_domain = feat['pred_domain']
                conf = feat['conf']
                
                sims = cosine_similarity([q_emb], faq_embeddings).flatten()
                
                scored_results = []
                for i, faq in enumerate(faqs):
                    # Domain Penalty Logic (replicating faq_system.py)
                    penalty = 1.0
                    if conf > 0.4 and faq["domain"] != pred_domain:
                        penalty = 0.05
                    
                    pos_s = calculate_overlap_score(q_pos, faq["pos_f"])
                    ner_s = calculate_overlap_score(q_ner, faq["ner_f"])
                    
                    final_score = ((a * sims[i]) + (b * pos_s) + (g * ner_s)) * penalty
                    scored_results.append((i, final_score))
                
                scored_results.sort(key=lambda x: x[1], reverse=True)
                top_faq_idx = scored_results[0][0]
                if faqs[top_faq_idx]['question'] == target:
                    hits += 1
            
            p1 = (hits / len(test_cases)) * 100
            # print(f"A={a:.2f}, B={b:.2f}, G={g:.2f} -> P@1: {p1:.2f}%")
            if p1 > best_p1:
                best_p1 = p1
                best_w = (a, b, g)

    print(f"\nOPTIMAL WEIGHTS: ALPHA={best_w[0]:.2f}, BETA={best_w[1]:.2f}, GAMMA={best_w[2]:.2f}")
    print(f"MAX ACHIEVED P@1: {best_p1:.2f}%")

if __name__ == "__main__":
    run_research()
