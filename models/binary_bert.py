import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
import psutil
import GPUtil

# Configuration
class Config:
    # Model
    MODEL_NAME = 'distilbert-base-uncased'
    NUM_LABELS = 2
    DROPOUT = 0.1
    
    # Training
    BATCH_SIZE = 64  # Per GPU batch size
    GRAD_ACCUM_STEPS = 2  # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS * N_GPUS
    MAX_EPOCHS = 7
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    
    # System
    USE_AMP = True  # Automatic Mixed Precision
    USE_MULTI_GPU = True  # DataParallel
    NUM_WORKERS = min(8, os.cpu_count() - 2)  # DataLoader workers
    
    # Checkpoints
    SAVE_DIR = './saved_models'
    SAVE_BEST_ONLY = True
    EARLY_STOPPING_PATIENCE = 3

# Initialize
def setup_training():
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Create save directory
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    # TensorBoard writer
    log_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    
    return device, n_gpu, writer

# Data Loading
def create_data_loaders(tokenized_dir='./data/processed'):
    train_data = torch.load(os.path.join(tokenized_dir, 'train_tokenized.pt'))
    val_data = torch.load(os.path.join(tokenized_dir, 'valid_tokenized.pt'))
    test_data = torch.load(os.path.join(tokenized_dir, 'test_tokenized.pt'))
    
    train_loader = DataLoader(
        train_data,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# Model Setup
def initialize_model(n_gpu):
    model = DistilBertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=Config.NUM_LABELS,
        dropout=Config.DROPOUT
    )
    
    if Config.USE_MULTI_GPU and n_gpu > 1:
        model = DataParallel(model)
    
    return model

# Training Components
def get_optimizer_scheduler(model, train_loader):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': Config.WEIGHT_DECAY,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=Config.LEARNING_RATE,
        eps=1e-8
    )
    
    total_steps = len(train_loader) * Config.MAX_EPOCHS // Config.GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler

# Training Loop
def train_model():
    device, n_gpu, writer = setup_training()
    train_loader, val_loader, _ = create_data_loaders()
    model = initialize_model(n_gpu)
    model.to(device)
    
    optimizer, scheduler = get_optimizer_scheduler(model, train_loader)
    scaler = GradScaler(enabled=Config.USE_AMP)
    criterion = nn.CrossEntropyLoss()
    
    # Training state
    best_val_loss = float('inf')
    epochs_no_improve = 0
    global_step = 0
    
    for epoch in range(Config.MAX_EPOCHS):
        # Training
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.MAX_EPOCHS}") as pbar:
            for step, batch in enumerate(pbar):
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[2]
                }
                
                with autocast(enabled=Config.USE_AMP):
                    outputs = model(**inputs)
                    loss = outputs.loss / Config.GRAD_ACCUM_STEPS
                
                scaler.scale(loss).backward()
                
                if (step + 1) % Config.GRAD_ACCUM_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                epoch_loss += loss.item() * Config.GRAD_ACCUM_STEPS
                pbar.set_postfix({'loss': loss.item() * Config.GRAD_ACCUM_STEPS})
                
                # Log metrics
                if global_step % 50 == 0:
                    writer.add_scalar('train/loss', loss.item(), global_step)
                    writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
                    
                    # Log GPU memory usage
                    if torch.cuda.is_available():
                        writer.add_scalar('system/gpu_mem', GPUtil.getGPUs()[0].memoryUsed, global_step)
                    
                    # Log CPU usage
                    writer.add_scalar('system/cpu_usage', psutil.cpu_percent(), global_step)
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, device)
        writer.add_scalar('val/loss', val_loss, global_step)
        writer.add_scalar('val/accuracy', val_acc, global_step)
        
        print(f"\nEpoch {epoch + 1} | Train Loss: {epoch_loss / len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_model(model, epoch, val_loss, val_acc, is_best=True)
        else:
            epochs_no_improve += 1
            if Config.EARLY_STOPPING_PATIENCE and epochs_no_improve >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping after {epoch + 1} epochs without improvement")
                break
    
    writer.close()
    return model

# Evaluation
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
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == batch[2]).sum().item()
            total += batch[2].size(0)
    
    return total_loss / len(data_loader), correct / total

# Model Saving
def save_model(model, epoch, val_loss, val_acc, is_best=False):
    # Get model from DataParallel
    if isinstance(model, DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model_state,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'config': vars(Config)
    }
    
    filename = f"distilbert_epoch{epoch + 1}_loss{val_loss:.4f}_acc{val_acc:.4f}.pt"
    filepath = os.path.join(Config.SAVE_DIR, filename)
    
    torch.save(checkpoint, filepath)
    print(f"Saved model checkpoint to {filepath}")
    
    if is_best:
        best_path = os.path.join(Config.SAVE_DIR, "best_model.pt")
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")

# Main Execution
if __name__ == "__main__":
    print("Starting training...")
    print(f"Configuration:\n{vars(Config)}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\nGPU Information:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel: {Config.USE_MULTI_GPU}")
    else:
        print("\nUsing CPU for training")
    
    trained_model = train_model()
    print("Training completed!")