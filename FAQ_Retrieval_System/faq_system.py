import sys
import json
import os
import numpy as np
import torch
from tqdm import tqdm

# Modular Imports
from config import (
    BASE_DIR, PREPROCESSOR_EXE, DATASET_FILE, GROUND_TRUTH_FILE,
    ALPHA, BETA, GAMMA, DOMAIN_KEYWORDS
)
from utils.dataset import parse_dataset
from utils.preprocessor import call_c_preprocessor
from utils.nlp_helpers import (
    robust_sentence_split, calculate_overlap_score, generate_abstract_answer
)
from classifier.domain import DomainClassifier
from engines.tfidf_engine import TFIDFEngine
from engines.sbert_engine import SBERTEngine
from engines.gensim_engine import GensimEngine
from engines.infersent_engine import InferSentEngine

def main():
    # Fix encoding for Windows consoles
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    print("Loading dataset...")
    faqs = parse_dataset(DATASET_FILE)
    print(f"Loaded {len(faqs)} FAQs.")
    
    print("Training domain classifier...")
    classifier = DomainClassifier()
    classifier.train(faqs)
    
    print("Preprocessing dataset using C preprocessor...")
    for faq in faqs:
        analysis = call_c_preprocessor(PREPROCESSOR_EXE, faq["question"])
        faq["tokens"] = " ".join(analysis["tokens"])
        faq["pos"] = analysis["pos"]
        faq["ner"] = analysis["ner"]
        
    print(f"Dataset preprocessed.")

    # Initialize ALL Engines
    engines = {}
    engine_types = ["tfidf", "sbert", "glove", "word2vec", "infersent"]
    
    print("\n--- Initializing Engines (This may take a few minutes) ---")
    for etype in engine_types:
        try:
            print(f"Loading {etype.upper()}...")
            if etype == "tfidf":
                engines[etype] = TFIDFEngine(faqs)
            elif etype == "sbert":
                engines[etype] = SBERTEngine(faqs)
            elif etype == "glove":
                engines[etype] = GensimEngine(faqs, 'glove')
            elif etype == "word2vec":
                engines[etype] = GensimEngine(faqs, 'word2vec')
            elif etype == "infersent":
                engines[etype] = InferSentEngine(faqs, base_dir=BASE_DIR)
            
            engines[etype].train()
        except Exception as e:
            print(f"Warning: Failed to load engine '{etype}': {e}")
            if etype in engines: del engines[etype]

    if not engines:
        print("Error: No engines could be initialized.")
        return

    print("\n" + "="*60)
    print("      🚀 MULTI-ENGINE FAQ COMPARISON SYSTEM      ")
    print("="*60)
    print(f"Active Engines: {', '.join(engines.keys()).upper()}")
    print("Commands: /exit, /help")
    
    while True:
        try:
            user_input = input("\nQuery > ").strip()
            if not user_input: continue
            if user_input.lower() in ['exit', 'quit', '/exit']: break

            if user_input.lower() == '/evaluate':
                if not os.path.exists(GROUND_TRUTH_FILE):
                    print(f"Error: {GROUND_TRUTH_FILE} not found. Run generator first.")
                    continue
                
                with open(GROUND_TRUTH_FILE, "r") as f:
                    test_cases = json.load(f)
                
                print(f"\n" + "🏁 STARTING SYSTEM EVALUATION ".center(80, "="))
                print(f"Dataset Size: {len(faqs)} FAQs")
                print(f"Test Set Size: {len(test_cases)} Paraphrased Queries")
                print("-" * 80)
                
                metrics = {name: {'hits': 0, 'rr_sum': 0.0} for name in engines.keys()}
                
                for case in tqdm(test_cases, desc="Evaluating Engines"):
                    t_query = case['query']
                    t_target = case['target']
                    
                    t_analysis = call_c_preprocessor(PREPROCESSOR_EXE, t_query)
                    t_q_tokens = " ".join(t_analysis["tokens"])
                    t_pred_domain, t_conf = classifier.predict(t_query)
                    
                    # Keyword Boosting Logic
                    for d, keywords in DOMAIN_KEYWORDS.items():
                        if any(k in t_query.lower() for k in keywords):
                            t_conf = max(t_conf, 0.95)
                            t_pred_domain = d.capitalize()
                            break

                    for name, engine in engines.items():
                        t_sims = engine.get_similarity(t_query if isinstance(engine, (SBERTEngine, InferSentEngine)) else t_q_tokens)
                        
                        all_e_scores = []
                        for idx, faq in enumerate(faqs):
                            d_penalty = 1.0
                            if t_conf > 0.4 and faq["domain"] != t_pred_domain:
                                d_penalty = 0.05
                            
                            sim_score = t_sims[idx]
                            pos_score = calculate_overlap_score(t_analysis["pos"], faq["pos"])
                            ner_score = calculate_overlap_score(t_analysis["ner"], faq["ner"])
                            e_final = ((ALPHA * sim_score) + (BETA * pos_score) + (GAMMA * ner_score)) * d_penalty
                            all_e_scores.append((idx, e_final))
                        
                        all_e_scores.sort(key=lambda x: x[1], reverse=True)
                        ranked_indices = [x[0] for x in all_e_scores]
                        
                        try:
                            target_faq_idx = next(i for i, f in enumerate(faqs) if f['question'] == t_target)
                            rank = ranked_indices.index(target_faq_idx) + 1
                            if rank == 1: metrics[name]['hits'] += 1
                            metrics[name]['rr_sum'] += (1.0 / rank)
                        except StopIteration: continue

                print(f"\n" + "📊 COMPARATIVE PERFORMANCE REPORT ".center(80, "="))
                print(f"{'ENGINE':<15} | {'P@1 (%)':<15} | {'MRR (Score)':<15}")
                print("-" * 80)
                for name in sorted(metrics.keys()):
                    p1 = (metrics[name]['hits'] / len(test_cases)) * 100
                    mrr = (metrics[name]['rr_sum'] / len(test_cases))
                    print(f"{name.upper():<15} | {p1:>8.2f}%       | {mrr:>10.4f}")
                print("=" * 80)
                continue

            # Standard Search Loop
            query = user_input
            pred_domain, confidence = classifier.predict(query)
            
            # Keyword Boosting
            query_lower = query.lower()
            for d, keywords in DOMAIN_KEYWORDS.items():
                if any(k in query_lower for k in keywords):
                    confidence = max(confidence, 0.95)
                    pred_domain = d.capitalize()
                    break

            analysis = call_c_preprocessor(PREPROCESSOR_EXE, query)
            q_tokens = " ".join(analysis["tokens"])
            
            print(f"\n" + "🔍 GLOBAL SEARCH RESULTS ".center(80, "="))
            print(f"Query: \"{query}\"")
            print(f"Intent Probability: {confidence:.2f} ({pred_domain.upper()})")

            engine_score_matrix = {} 
            for name, engine in engines.items():
                sims = engine.get_similarity(query if isinstance(engine, (SBERTEngine, InferSentEngine)) else q_tokens)
                
                final_engine_scores = np.zeros(len(faqs))
                for idx, faq in enumerate(faqs):
                    domain_penalty = 1.0
                    if confidence > 0.4 and faq["domain"] != pred_domain:
                        domain_penalty = 0.05
                    
                    sim_score = sims[idx]
                    pos_score = calculate_overlap_score(analysis["pos"], faq["pos"])
                    ner_score = calculate_overlap_score(analysis["ner"], faq["ner"])
                    final_score = ((ALPHA * sim_score) + (BETA * pos_score) + (GAMMA * ner_score)) * domain_penalty
                    final_engine_scores[idx] = final_score
                
                engine_score_matrix[name] = final_engine_scores

            all_scores_stacked = np.array(list(engine_score_matrix.values()))
            max_scores = np.max(all_scores_stacked, axis=0)
            top_indices = np.argsort(max_scores)[-3:][::-1]
            
            print(f"\n--- TOP 3 MATCHES (MIXED DOMAINS) ---")
            top_results = []
            for i, idx in enumerate(top_indices):
                score = max_scores[idx]
                faq = faqs[idx]
                top_results.append((idx, score))
                print(f"{i+1}. [{faq['domain']}] {faq['question']} (Confidence: {score:.3f})")
            
            # Stage 6: Synthesis
            summary = generate_abstract_answer(top_results, faqs)
            print(f"\n--- SYNTHESIZED SUMMARY ---")
            print(summary)
            
            if len(top_indices) > 0:
                top_idx = top_indices[0]
                top_faq = faqs[top_idx]
                print(f"\n--- TOP MATCH DETAILS ---")
                print(f"Question: {top_faq['question']}")
                print(f"Answer:   {top_faq['answer']}")
                
                print(f"\n[ Confidence Breakdown for Top Match ]")
                for name, scores in engine_score_matrix.items():
                    print(f"- {name.upper():<10}: {scores[top_idx]:.4f}")
                
                best_overall = max_scores[top_idx]
                agreed_engines = [name for name, scores in engine_score_matrix.items() if scores[top_idx] >= 0.9 * best_overall]
                if agreed_engines:
                    print(f"\n(Verified by: {', '.join(agreed_engines)})")
            
            print("=" * 80)

        except KeyboardInterrupt: break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
