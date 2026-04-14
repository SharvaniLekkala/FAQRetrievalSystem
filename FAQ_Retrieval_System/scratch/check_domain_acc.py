import json
import os
import sys
import numpy as np
from tqdm import tqdm

# Local imports
sys.path.append(os.getcwd())
from classifier.domain import DomainClassifier
from utils.dataset import parse_dataset
from config import DATASET_FILE, GROUND_TRUTH_FILE

def check_domain_accuracy():
    print("Loading dataset...")
    faqs = parse_dataset(DATASET_FILE)
    
    print("Loading ground truth...")
    with open(GROUND_TRUTH_FILE, "r") as f:
        test_cases = json.load(f)
    
    # We need the ground truth labels for domains.
    # We can infer them from the target questions in the dataset.
    question_to_domain = {f["question"]: f["domain"] for f in faqs}
    
    print("Training Domain Classifier...")
    classifier = DomainClassifier()
    classifier.train(faqs)
    
    correct = 0
    total = 0
    
    print("Checking domain classification on test queries...")
    for case in test_cases:
        query = case['query']
        target = case['target']
        true_domain = question_to_domain.get(target)
        
        if not true_domain:
            continue
            
        pred_domain, conf = classifier.predict(query)
        
        if pred_domain.lower() == true_domain.lower():
            correct += 1
        else:
            print(f"MISCLASSIFIED: Query='{query}' | Pred='{pred_domain}' | True='{true_domain}' | Conf={conf:.2f}")
        
        total += 1
        
    acc = (correct / total) * 100
    print(f"\nDomain Classifier Accuracy on Test Queries: {acc:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    check_domain_accuracy()
