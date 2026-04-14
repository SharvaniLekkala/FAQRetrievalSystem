import json
import os
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def train():
    # Load augmented data
    train_data_path = 'fine_tuning/augmented_train.json'
    if not os.path.exists(train_data_path):
        print(f"Error: {train_data_path} not found.")
        return

    with open(train_data_path, 'r') as f:
        train_data = json.load(f)

    # Convert to InputExample objects
    train_examples = []
    for item in train_data:
        train_examples.append(InputExample(texts=[item['query'], item['target']]))

    # Model definition - using the stronger MPNet base
    model_name = 'all-mpnet-base-v2'
    print(f"Loading base model: {model_name}")
    model = SentenceTransformer(model_name)

    # DataLoader
    # Very small batch size for CPU training to manage system memory
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)

    # Loss function - MultipleNegativesRankingLoss is excellent for retrieval
    # It treats (query, target) as positive and other targets in the batch as negatives
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Output path
    output_path = 'fine_tuned_sbert_v2'
    os.makedirs(output_path, exist_ok=True)

    # Training
    # We use 10 epochs for the massive Round 3 dataset to ensure deep specialization
    print("Starting Phase 3 Deep Fine-tuning (CPU)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=10,
        warmup_steps=10,
        output_path=output_path,
        show_progress_bar=True
    )

    print(f"Fine-tuning complete! Model saved to {output_path}")

if __name__ == "__main__":
    train()
