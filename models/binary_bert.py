import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import (
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np

# Fix for PyTorch 2.6+ serialization
torch.serialization.add_safe_globals([TensorDataset])

class Config:
    # Model
    MODEL_NAME = 'distilbert-base-uncased'
    NUM_LABELS = 2
    DROPOUT = 0.1
    
    # Training
    BATCH_SIZE = 32
    GRAD_ACCUM_STEPS = 4
    MAX_EPOCHS = 5
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 100
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    
    # System
    USE_AMP = False  # Disabled for CPU
    NUM_WORKERS = min(4, os.cpu_count() - 1)
    
    # Checkpoints
    SAVE_DIR = './saved_models'
    SAVE_BEST_ONLY = True
    EARLY_STOPPING_PATIENCE = 2

def verify_environment():
    """Check all requirements before starting training"""
    print("\n🚀 Starting pre-flight checks...")
    
    # Verify PyTorch installation
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")
    except ImportError:
        sys.exit("❌ PyTorch not installed. Please install PyTorch first.")
    
    # Verify CUDA (will fall back to CPU if not available)
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            print(f"✅ CUDA available (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("ℹ️ CUDA not available, falling back to CPU")
    except RuntimeError as e:
        print(f"⚠️ CUDA check failed: {e}, falling back to CPU")
        device = torch.device('cpu')
    
    # Verify data directory
    data_dir = './data/processed'
    required_files = ['train_tokenized.pt', 'valid_tokenized.pt']
    missing_files = []
    for f in required_files:
        if not os.path.exists(os.path.join(data_dir, f)):
            missing_files.append(f)
    if missing_files:
        sys.exit(f"❌ Missing required data files: {missing_files}")
    print("✅ All data files found")
    
    # Verify save directory
    try:
        os.makedirs(Config.SAVE_DIR, exist_ok=True)
        test_file = os.path.join(Config.SAVE_DIR, 'test_write.tmp')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("✅ Save directory is writable")
    except Exception as e:
        sys.exit(f"❌ Cannot write to save directory: {e}")
    
    print("🎉 All pre-flight checks passed!\n")
    return device

def load_data_safely():
    """Load data with comprehensive error handling"""
    data_dir = './data/processed'
    files = {
        'train': 'train_tokenized.pt',
        'valid': 'valid_tokenized.pt',
        'test': 'test_tokenized.pt'
    }
    
    loaded_data = {}
    for name, filename in files.items():
        try:
            filepath = os.path.join(data_dir, filename)
            loaded_data[name] = torch.load(filepath, weights_only=False)
            print(f"✅ Loaded {filename} successfully")
        except Exception as e:
            sys.exit(f"❌ Failed to load {filename}: {e}")
    
    return loaded_data['train'], loaded_data['valid'], loaded_data['test']

def initialize_model(device):
    """Initialize model with verification"""
    try:
        print("\nInitializing model...")
        model = DistilBertForSequenceClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=Config.NUM_LABELS,
            dropout=Config.DROPOUT
        )
        model.to(device)
        print("✅ Model initialized successfully")
        return model
    except Exception as e:
        sys.exit(f"❌ Model initialization failed: {e}")

def create_data_loaders(train_data, val_data, test_data, device):
    """Create data loaders with validation"""
    try:
        print("\nCreating data loaders...")
        train_loader = DataLoader(
            train_data,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=device.type == 'cuda'
        )
        val_loader = DataLoader(
            val_data,
            batch_size=Config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=device.type == 'cuda'
        )
        test_loader = DataLoader(
            test_data,
            batch_size=Config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=device.type == 'cuda'
        )
        print("✅ Data loaders created successfully")
        return train_loader, val_loader, test_loader
    except Exception as e:
        sys.exit(f"❌ Data loader creation failed: {e}")

def train_model():
    # Verify everything before starting
    device = verify_environment()
    
    # Load data
    train_data, val_data, test_data = load_data_safely()
    
    # Initialize components
    model = initialize_model(device)
    train_loader, val_loader, _ = create_data_loaders(train_data, val_data, test_data, device)
    
    # Training setup
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.WARMUP_STEPS,
        num_training_steps=len(train_loader) * Config.MAX_EPOCHS // Config.GRAD_ACCUM_STEPS
    )
    
    # Training state
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print("\n🏋️ Starting training...")
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
            save_model(model, optimizer, epoch, val_loss, val_acc, is_best=True)
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

def save_model(model, optimizer, epoch, val_loss, val_acc, is_best=False):
    try:
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
            print(f"💾 Saved best model to {best_path}")
    except Exception as e:
        print(f"⚠️ Failed to save model: {e}")

if __name__ == "__main__":
    print("Starting training with configuration:")
    print("\n".join(f"{k}: {v}" for k, v in vars(Config).items() if not k.startswith('__')))
    
    try:
        trained_model = train_model()
        print("\n🎉 Training completed successfully!")
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        sys.exit(1)