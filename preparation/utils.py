import os
import pandas as pd
import csv
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ----------------------------
# 🛠️ Dataset Utilities
# ----------------------------

def load_dataset(file_path, delimiter):
    """Loads a dataset from CSV, ensuring correct data types."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file '{file_path}' not found.")

    try:
        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            quoting=csv.QUOTE_MINIMAL,  # Ensures quoted strings with delimiters inside don't break parsing
            on_bad_lines="skip",  # Skips malformed rows
            encoding="utf-8"
        )
    except Exception as e:
        print(f"❌ CSV Parsing Error: {e}")
        raise

    df['label'] = df['label'].astype(int)
    return df


def split_dataset(df):
    """Splits the dataset into train, validation, and test sets."""
    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)
    
    return train_df, val_df, test_df

def tokenize_function(examples, tokenizer):
    """Tokenizes dataset inputs."""
    return tokenizer(
        [f"Title: {title} URL: {url} Sentence: {sentence}" 
         for title, url, sentence in zip(examples['title'], examples['url'], examples['sentence'])],
        padding="max_length",
        truncation=True,
        max_length=512
    )

def compute_metrics(pred):
    """Computes accuracy, precision, recall, and F1-score."""
    logits = pred.predictions
    labels = pred.label_ids
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def get_model_and_tokenizer(model_choice, resume):
    """Loads the tokenizer and model. If `resume` is True, loads from fine-tuned model directory."""
    if resume:
        model_path = f"fine_tuned_{model_choice}"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fine-tuned model '{model_path}' not found.")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        if model_choice == "distilbert":
            model_name = "distilbert-base-uncased"
        elif model_choice == "minilm":
            model_name = "microsoft/MiniLM-L12-H384-uncased"
        else:
            raise ValueError(f"Invalid model choice '{model_choice}'. Choose 'distilbert' or 'minilm'.")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    return tokenizer, model
