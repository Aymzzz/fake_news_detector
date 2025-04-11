import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report

# Directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models/trained/distilbert_binary.pth")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# Load tokenized datasets
train_dataset = torch.load(os.path.join(PROCESSED_DIR, "train_tokenized.pt"))
valid_dataset = torch.load(os.path.join(PROCESSED_DIR, "valid_tokenized.pt"))
test_dataset = torch.load(os.path.join(PROCESSED_DIR, "test_tokenized.pt"))

# Dataloaders
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
model.to(device)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 6
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Loss
loss_fn = nn.CrossEntropyLoss()

# Training loop
best_val_loss = float("inf")
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch in loop:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0
    preds, targets = [], []

    with torch.no_grad():
        for batch in valid_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()
            preds.extend(torch.argmax(outputs.logits, dim=1).tolist())
            targets.extend(labels.tolist())

    avg_val_loss = total_val_loss / len(valid_loader)
    print(f"🔍 Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✅ Best model saved to: {MODEL_SAVE_PATH}")

# Final Evaluation
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()
preds, targets = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        preds.extend(torch.argmax(outputs.logits, dim=1).tolist())
        targets.extend(labels.tolist())

print("\n📊 Classification Report (Test Set):")
print(classification_report(targets, preds, target_names=["False", "True"]))
