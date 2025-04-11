import os
import re
import json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from tqdm import tqdm
from datetime import datetime

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column headers
COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job", "state", "party",
    "barely_true_counts", "half_true_counts", "mostly_true_counts", "false_counts",
    "pants_on_fire_counts", "context"
]

def clean_text(text):
    """Enhanced text cleaning with thorough type checking"""
    if text is None:
        return "[EMPTY]"
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return "[EMPTY]"
    text = text.strip()
    if not text:
        return "[EMPTY]"
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else "[EMPTY]"

def debug_text_issues(df):
    """Identify and log problematic text entries"""
    issues = []
    for idx, row in df.iterrows():
        text = row['statement']
        if not isinstance(text, str):
            issues.append(f"Row {idx}: Non-string type {type(text)} - Value: {text}")
        elif not text.strip():
            issues.append(f"Row {idx}: Empty string")
    
    if issues:
        debug_file = os.path.join(OUTPUT_DIR, "text_issues.log")
        with open(debug_file, 'w') as f:
            f.write("\n".join(issues))
        print(f"⚠️ Found {len(issues)} text issues - saved to {debug_file}")
    else:
        print("✅ No text issues found")

def load_and_combine_data():
    """Load and combine all TSV files with validation"""
    print("Loading and combining dataset splits...")
    dfs = []
    for split in ['train', 'valid', 'test']:
        filepath = os.path.join(RAW_DIR, f"{split}.tsv")
        try:
            df = pd.read_csv(filepath, sep='\t', header=None, names=COLUMNS)
            dfs.append(df)
            print(f"Loaded {len(df)} {split} samples")
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            raise
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total samples loaded: {len(combined)}")
    return combined

def preprocess_data(df):
    """Clean text and create binary labels with validation"""
    print("Preprocessing data...")
    
    # Make copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Clean text with progress
    tqdm.pandas(desc="Cleaning text")
    df["statement"] = df["statement"].progress_apply(clean_text)
    
    # Debug any remaining issues
    debug_text_issues(df)
    
    # Create binary labels
    true_labels = ["true", "mostly-true", "half-true"]
    false_labels = ["false", "barely-true", "pants-fire"]
    df["binary_label"] = df["label"].apply(lambda x: 1 if x in true_labels else 0)
    
    return df[["statement", "binary_label"]]

def balance_and_split_data(df):
    """Balance classes and create train/valid/test splits"""
    print("Balancing and splitting data...")
    
    # Balance dataset
    value_counts = df["binary_label"].value_counts()
    print(f"Class distribution before balancing:\n{value_counts}")
    min_class_size = value_counts.min()
    
    df_balanced = pd.concat([
        df[df["binary_label"] == 1].sample(min_class_size, random_state=42),
        df[df["binary_label"] == 0].sample(min_class_size, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data
    train_val_df, test_df = train_test_split(
        df_balanced, test_size=0.15, stratify=df_balanced["binary_label"], random_state=42
    )
    train_df, valid_df = train_test_split(
        train_val_df, test_size=0.15, stratify=train_val_df["binary_label"], random_state=42
    )
    
    print("\nFinal split sizes:")
    print(f"Train samples: {len(train_df)}")
    print(f"Valid samples: {len(valid_df)}")
    print(f"Test samples: {len(test_df)}")
    
    return train_df, valid_df, test_df

def save_tsvs(train_df, valid_df, test_df):
    """Save the processed TSV files"""
    print("\nSaving processed TSV files...")
    train_df.to_csv(os.path.join(OUTPUT_DIR, "binary_train.tsv"), sep='\t', index=False, header=False)
    valid_df.to_csv(os.path.join(OUTPUT_DIR, "binary_valid.tsv"), sep='\t', index=False, header=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "binary_test.tsv"), sep='\t', index=False, header=False)

def validate_text_data(text_list):
    """Ensure all text entries are valid strings"""
    validated = []
    for i, text in enumerate(text_list):
        if not isinstance(text, str):
            print(f"Found non-string at index {i}: {type(text)} - {text}")
            text = str(text) if text is not None else "[EMPTY]"
        validated.append(text)
    return validated

def tokenize_and_save(df, filename):
    """Tokenize text and save as PyTorch tensors with extensive validation"""
    print(f"\nTokenizing {filename.replace('_tokenized.pt', '')} data...")
    
    # Extract and validate statements
    statements = df["statement"].tolist()
    statements = validate_text_data(statements)
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    # Tokenize in batches with error handling
    try:
        encodings = tokenizer(
            statements,
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
    except Exception as e:
        print(f"❌ Tokenization failed: {str(e)}")
        
        # Save problematic data for debugging
        debug_data = {
            "error": str(e),
            "sample_statements": statements[:100],  # Save first 100 for inspection
            "shape": df.shape,
            "columns": list(df.columns)
        }
        debug_file = os.path.join(OUTPUT_DIR, f"tokenization_error_{filename}.json")
        with open(debug_file, 'w') as f:
            json.dump(debug_data, f, indent=2)
        print(f"Saved debug info to {debug_file}")
        raise

def save_metadata(train_df, valid_df, test_df):
    """Save comprehensive metadata"""
    metadata = {
        "created_at": str(datetime.now()),
        "dataset_info": {
            "total_samples": len(train_df) + len(valid_df) + len(test_df),
            "train_samples": len(train_df),
            "valid_samples": len(valid_df),
            "test_samples": len(test_df),
            "class_balance": {
                "train": {
                    "true": int(train_df["binary_label"].sum()),
                    "false": len(train_df) - int(train_df["binary_label"].sum())
                },
                "valid": {
                    "true": int(valid_df["binary_label"].sum()),
                    "false": len(valid_df) - int(valid_df["binary_label"].sum())
                },
                "test": {
                    "true": int(test_df["binary_label"].sum()),
                    "false": len(test_df) - int(test_df["binary_label"].sum())
                }
            }
        },
        "processing": {
            "text_cleaning": "enhanced",
            "tokenizer": "distilbert-base-uncased",
            "max_length": 256,
            "random_state": 42
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("💾 Saved metadata to metadata.json")

def main():
    try:
        # Load and process data
        df = load_and_combine_data()
        df = preprocess_data(df)
        train_df, valid_df, test_df = balance_and_split_data(df)
        
        # Save processed data
        save_tsvs(train_df, valid_df, test_df)
        
        # Tokenize and save datasets
        tokenize_and_save(train_df, "train_tokenized.pt")
        tokenize_and_save(valid_df, "valid_tokenized.pt")
        tokenize_and_save(test_df, "test_tokenized.pt")
        
        # Save metadata
        save_metadata(train_df, valid_df, test_df)
        
        print("\n✅ All processing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()