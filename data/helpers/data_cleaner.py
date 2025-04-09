# prepare_embeddings.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import DistilBertTokenizer

COLUMN_NAMES = [
    "id", "label", "statement", "subject", "speaker", "speaker_job",
    "state_info", "party_affiliation", "barely_true_count", "false_count",
    "half_true_count", "mostly_true_count", "pants_on_fire_count", "context"
]

CREDIBILITY_COLS = [
    'barely_true_count', 'false_count', 'half_true_count',
    'mostly_true_count', 'pants_on_fire_count'
]

LABEL_MAP = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}

DATA_PATHS = {
    "train": "data/raw/train.tsv",
    "valid": "data/raw/valid.tsv",
    "test":  "data/raw/test.tsv"
}

SAVE_DIR = "data/processed"
os.makedirs(SAVE_DIR, exist_ok=True)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def load_dataset(path):
    """Load a TSV dataset with predefined column names."""
    return pd.read_csv(path, sep='\t', names=COLUMN_NAMES)

def preprocess_dataframe(df):
    """Fill missing values, convert counts to integers, and compute speaker credibility."""
    df.fillna("-", inplace=True)

    for col in CREDIBILITY_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    df['speaker_credibility'] = df.apply(compute_credibility, axis=1)
    df['party_affiliation'] = df['party_affiliation'].fillna('Unknown')
    df = pd.get_dummies(df, columns=['party_affiliation'], prefix='party')

    return df

def compute_credibility(row):
    """Compute a credibility score based on truth and false counts."""
    pos = row['mostly_true_count'] + row['half_true_count']
    neg = row['false_count'] + row['pants_on_fire_count']
    return pos - neg

def tokenize_statements(statements, desc="Tokenizing"):
    """Tokenize a list of statements using DistilBERT tokenizer."""
    return tokenizer(list(tqdm(statements, desc=desc)), truncation=True, padding=True)

# Encode labels
def encode_labels(df, label_map):
    return df['label'].map(label_map).astype(int).values

def save_embeddings(encodings, base_filename):
    """Save input_ids and attention_mask arrays as .npy files."""
    input_ids_path = os.path.join(SAVE_DIR, f"{base_filename}_input_ids.npy")
    attention_mask_path = os.path.join(SAVE_DIR, f"{base_filename}_attention_mask.npy")

    np.save(input_ids_path, encodings['input_ids'])
    np.save(attention_mask_path, encodings['attention_mask'])

    print(f"Saved {base_filename} input_ids to: {input_ids_path}")
    print(f"Saved {base_filename} attention_mask to: {attention_mask_path}")

def save_labels(labels, base_filename):
    labels_path = os.path.join(SAVE_DIR, f"{base_filename}_labels.npy")
    np.save(labels_path, labels)
    print(f"Saved {base_filename} labels to: {labels_path}")

def main():
    print("Loading datasets...")
    train_df = load_dataset(DATA_PATHS["train"])
    valid_df = load_dataset(DATA_PATHS["valid"])
    test_df = load_dataset(DATA_PATHS["test"])

    print("Preprocessing datasets...")
    train_df = preprocess_dataframe(train_df)
    valid_df = preprocess_dataframe(valid_df)
    test_df = preprocess_dataframe(test_df)

    print("\nSample processed training data:")
    print(train_df[['statement', 'speaker_credibility']].head())

    print("\nTokenizing statements...")
    train_enc = tokenize_statements(train_df['statement'], "Train")
    valid_enc = tokenize_statements(valid_df['statement'], "Validation")
    test_enc = tokenize_statements(test_df['statement'], "Test")

    print("\nSaving tokenized embeddings...")
    save_embeddings(train_enc, "train")
    save_embeddings(valid_enc, "valid")
    save_embeddings(test_enc, "test")

    print("\nEncoding and saving labels...")
    train_labels = encode_labels(train_df, LABEL_MAP)
    valid_labels = encode_labels(valid_df, LABEL_MAP)
    test_labels = encode_labels(test_df, LABEL_MAP)

    save_labels(train_labels, "train")
    save_labels(valid_labels, "valid")
    save_labels(test_labels, "test")

    print("\nAll steps completed successfully!")

if __name__ == "__main__":
    main()
