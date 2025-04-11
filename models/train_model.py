import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertForSequenceClassification, get_scheduler
from torch.optim import AdamW 
from sklearn.metrics import classification_report
from tqdm import tqdm

# ============================
# Set paths relative to script
# ============================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # This script's folder (models)
PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, ".."))  # Move up to root
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'multy-output')
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'models', 'trained')

# ===============
# Device Setup
# ===============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ===========
# Parameters
# ===========
MODEL_NAME = 'distilbert-base-uncased'
BATCH_SIZE = 32
EPOCHS = 7

# ===============
# Load functions
# ===============
def load_data(split):
    """Load processed features and labels from multy-output directory."""
    input_ids = np.load(os.path.join(DATA_PROCESSED_DIR, f"{split}_input_ids.npy"))
    attention_mask = np.load(os.path.join(DATA_PROCESSED_DIR, f"{split}_attention_mask.npy"))
    labels = np.load(os.path.join(DATA_PROCESSED_DIR, f"{split}_labels.npy"))

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long)
    )

def create_dataloader(inputs, masks, labels, sampler_type='random'):
    dataset = TensorDataset(inputs, masks, labels)
    sampler = RandomSampler(dataset) if sampler_type == 'random' else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)

# ==========
# Training
# ==========
def train():
    print("Loading data...")
    train_inputs, train_masks, train_labels = load_data("train")
    valid_inputs, valid_masks, valid_labels = load_data("valid")

    train_loader = create_dataloader(train_inputs, train_masks, train_labels, 'random')
    valid_loader = create_dataloader(valid_inputs, valid_masks, valid_labels, 'sequential')

    print("Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Training loop
    print("Starting training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in loop:
            b_input_ids, b_attention_mask, b_labels = [x.to(DEVICE) for x in batch]
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

    # Evaluation
    print("\nEvaluating on validation set...")
    model.eval()
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in valid_loader:
            b_input_ids, b_attention_mask, b_labels = [x.to(DEVICE) for x in batch]
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, axis=1).cpu().numpy())
            true_labels.extend(b_labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(true_labels, preds, target_names=[
        "pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"
    ]))

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_SAVE_DIR, 'distilbert_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ Model saved at: {model_save_path}")

# ==========
# Run
# ==========
if __name__ == "__main__":
    train()