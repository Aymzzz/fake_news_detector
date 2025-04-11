import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DataParallel
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import (
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from datetime import datetime
import numpy as np

# Fix for PyTorch 2.6+ serialization
torch.serialization.add_safe_globals([TensorDataset])

class Config:
    # Model
    MODEL_NAME = 'distilbert-base-uncased'
    NUM_LABELS = 2
    DROPOUT = 0.1
    
    # Training
    BATCH_SIZE = 32  # Reduced for CPU training
    GRAD_ACCUM_STEPS = 4  # Compensate for smaller batch size
    MAX_EPOCHS = 5  # Reduced for initial testing
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 100  # Reduced for fewer epochs
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    
    # System
    USE_AMP = False  # Disabled for CPU
    USE_MULTI_GPU = False  # Disabled for CPU
    NUM_WORKERS = min(4, os.cpu_count() - 1)  # Reduced for CPU
    
    # Checkpoints
    SAVE_DIR = './saved_models'
    SAVE_BEST_ONLY = True
    EARLY_STOPPING_PATIENCE = 2  # Reduced for fewer epochs

def setup_environment():
    # Handle CUDA initialization error gracefully
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except RuntimeError as e:
        print(f"CUDA initialization error: {e}")
        device = torch.device('cpu')
        n_gpu = 0
    
    print(f"\nUsing device: {device}")
    print(f"Number of workers: {Config.NUM_WORKERS}")
    
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    return device, n_gpu

def load_data_safely(tokenized_dir='./data/processed'):
    """Handle the PyTorch serialization issue with weights_only=False"""
    try:
        # First try with weights_only=True
        train_data = torch.load(
            os.path.join(tokenized_dir, 'train_tokenized.pt'),
            weights_only=True
        )
        val_data = torch.load(
            os.path.join(tokenized_dir, 'valid_tokenized.pt'),
            weights_only=True
        )
        test_data = torch.load(
            os.path.join(tokenized_dir, 'test_tokenized.pt'),
            weights_only=True
        )
    except Exception as e:
        print(f"Standard loading failed: {e}")
        print("Attempting with weights_only=False (only use with trusted data)")
        train_data = torch.load(
            os.path.join(tokenized_dir, 'train_tokenized.pt'),
            weights_only=False
        )
        val_data = torch.load(
            os.path.join(tokenized_dir, 'valid_tokenized.pt'),
            weights_only=False
        )
        test_data = torch.load(
            os.path.join(tokenized_dir, 'test_tokenized.pt'),
            weights_only=False
        )
    
    return train_data, val_data, test_data

def create_data_loaders():
    train_data, val_data, test_data = load_data_safely()
    
    train_loader = DataLoader(
        train_data,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False  # Disabled for CPU
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader

def initialize_model(device):
    model = DistilBertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=Config.NUM_LABELS,
        dropout=Config.DROPOUT
    )
    model.to(device)
    return model

def train_model():
    device, n_gpu = setup_environment()
    train_loader, val_loader, _ = create_data_loaders()
    model = initialize_model(device)
    
    # Training components
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.WARMUP_STEPS,
        num_training_steps=len(train_loader) * Config.MAX_EPOCHS // Config.GRAD_ACCUM_STEPS
    )
    
    # Training state
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.MAX_EPOCHS}") as pbar:
            for step, batch in enumerate(pbar):
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[2]
                }
                
                outputs = model(**inputs)
                loss = outputs.loss / Config.GRAD_ACCUM_STEPS
                loss.backward()
                
                if (step + 1) % Config.GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * Config.GRAD_ACCUM_STEPS
                pbar.set_postfix({'loss': loss.item() * Config.GRAD_ACCUM_STEPS})
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"\nEpoch {epoch + 1} | Train Loss: {epoch_loss / len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_model(model, epoch, val_loss, val_acc, is_best=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping after {epoch + 1} epochs without improvement")
                break
    
    return model

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]
            }
            
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == batch[2]).sum().item()
            total += batch[2].size(0)
    
    return total_loss / len(data_loader), correct / total

def save_model(model, epoch, val_loss, val_acc, is_best=False):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'config': vars(Config)
    }
    
    filename = f"distilbert_epoch{epoch + 1}_loss{val_loss:.4f}_acc{val_acc:.4f}.pt"
    filepath = os.path.join(Config.SAVE_DIR, filename)
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_path = os.path.join(Config.SAVE_DIR, "best_model.pt")
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")

if __name__ == "__main__":
    print("Starting training with configuration:")
    print("\n".join(f"{k}: {v}" for k, v in vars(Config).items() if not k.startswith('__')))
    
    try:
        trained_model = train_model()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise