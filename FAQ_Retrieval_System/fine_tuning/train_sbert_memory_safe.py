import json
import os
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader, Dataset

# Memory-safe Dataset class
class FAQDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return InputExample(texts=[item['query'], item['target']])

def train():
    train_data_path = 'fine_tuning/augmented_train.json'
    if not os.path.exists(train_data_path):
        print(f"Error: {train_data_path} not found.")
        return

    # Using a custom dataset to avoid keeping all objects in a single list if possible
    dataset = FAQDataset(train_data_path)
    
    # Model definition - switching to a lightweight but powerful L12 model
    model_name = 'all-MiniLM-L12-v2'
    print(f"Loading base model: {model_name}")
    model = SentenceTransformer(model_name)

    # DataLoader - Very small batch size for CPU
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=4)

    # Loss function
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    output_path = 'fine_tuned_sbert_v2'
    os.makedirs(output_path, exist_ok=True)

    print("Starting Phase 3 Deep Fine-tuning (Memory-Safe)...")
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=10,
            warmup_steps=100,
            output_path=output_path,
            show_progress_bar=True
        )
        print(f"Fine-tuning complete! Model saved to {output_path}")
    except Exception as e:
        print(f"FATAL ERROR during training: {e}")

if __name__ == "__main__":
    train()
