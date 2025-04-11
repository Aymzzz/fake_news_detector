import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load original splits
train_df = pd.read_csv(os.path.join(RAW_DIR, "train.tsv"), sep='\t', header=None)
valid_df = pd.read_csv(os.path.join(RAW_DIR, "valid.tsv"), sep='\t', header=None)
test_df = pd.read_csv(os.path.join(RAW_DIR, "test.tsv"), sep='\t', header=None)

# Combine all
df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
df.columns = [
    "id", "label", "statement", "subject", "speaker", "job", "state", "party",
    "barely_true_counts", "half_true_counts", "mostly_true_counts", "false_counts",
    "pants_on_fire_counts", "context"
]

# Reduce labels to binary
true_labels = ["true", "mostly-true", "half-true"]
false_labels = ["false", "barely-true", "pants-fire"]
df["binary_label"] = df["label"].apply(lambda x: 1 if x in true_labels else 0)

# Keep only the relevant fields
df = df[["statement", "binary_label"]]

# Balance dataset
min_class_size = df["binary_label"].value_counts().min()
df_balanced = pd.concat([
    df[df["binary_label"] == 1].sample(min_class_size, random_state=42),
    df[df["binary_label"] == 0].sample(min_class_size, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

# Split
train_val_df, test_df = train_test_split(df_balanced, test_size=0.15, stratify=df_balanced["binary_label"], random_state=42)
train_df, valid_df = train_test_split(train_val_df, test_size=0.15, stratify=train_val_df["binary_label"], random_state=42)

# Save TSVs
train_df.to_csv(os.path.join(OUTPUT_DIR, "binary_train.tsv"), sep='\t', index=False, header=False)
valid_df.to_csv(os.path.join(OUTPUT_DIR, "binary_valid.tsv"), sep='\t', index=False, header=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "binary_test.tsv"), sep='\t', index=False, header=False)

print("✅ TSVs saved. Now tokenizing with DistilBERT...")

# Tokenization function
def tokenize_and_save(df, filename):
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    encodings = tokenizer(
        df["statement"].tolist(),
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )
    labels = torch.tensor(df["binary_label"].tolist())
    dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"], encodings["attention_mask"], labels
    )
    torch.save(dataset, os.path.join(OUTPUT_DIR, filename))
    print(f"💾 Saved tokenized dataset to {filename}")

# Tokenize and save all three
tokenize_and_save(train_df, "train_tokenized.pt")
tokenize_and_save(valid_df, "valid_tokenized.pt")
tokenize_and_save(test_df, "test_tokenized.pt")

print("✅ All tokenized datasets saved successfully.")
