import argparse
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ----------------------------
# 🛠️ Utility Functions
# ----------------------------

def load_dataset(file_path):
    """Loads the dataset from CSV and ensures correct data types."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file '{file_path}' not found. Ensure it exists in the project directory.")

    df = pd.read_csv(file_path)
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

def get_model_and_tokenizer(model_choice):
    """Loads the specified model and tokenizer."""
    if model_choice == "distilbert":
        model_name = "distilbert-base-uncased"
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    elif model_choice == "minilm":
        model_name = "microsoft/MiniLM-L12-H384-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    else:
        raise ValueError(f"Invalid model choice '{model_choice}'. Choose 'distilbert' or 'minilm'.")
    
    return tokenizer, model

# ----------------------------
# 🚀 Main Training Function
# ----------------------------

def train_model(model_choice, dataset_path="labeled_dataset.csv"):
    """Trains and evaluates a model based on the specified choice."""

    print(f"\n🚀 Training with model: {model_choice}\n")

    # Step 1: Load and split dataset
    df = load_dataset(dataset_path)
    train_df, val_df, test_df = split_dataset(df)

    # Step 2: Load tokenizer and model
    tokenizer, model = get_model_and_tokenizer(model_choice)

    # Step 3: Convert to Dataset object
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Step 4: Tokenize dataset
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Set dataset format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Step 5: Define training arguments
    training_args = TrainingArguments(
        output_dir=f'./results_{model_choice}',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir=f'./logs_{model_choice}',
        logging_steps=10,
        metric_for_best_model="eval_accuracy"
    )

    # Step 6: Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Step 7: Train the model
    print("\n🚀 Starting model training...\n")
    trainer.train()

    # Step 8: Save the model and tokenizer
    save_path = f"fine_tuned_{model_choice}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n✅ Model and tokenizer saved to '{save_path}'\n")

    # Step 9: Evaluate the model
    print("\n🔍 Evaluating on test dataset...\n")
    test_results = trainer.evaluate(test_dataset)

    # Display results
    print("\n📊 Test Dataset Evaluation:")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")

    # Step 10: Save results
    os.makedirs("evaluation", exist_ok=True)
    results_file = f"evaluation/{model_choice}_performance.csv"
    pd.DataFrame([test_results]).to_csv(results_file, index=False)
    
    print(f"\n📁 Test results saved to: {results_file}\n")
    print("\n🎉 Training process completed successfully!")

# ----------------------------
# 🔹 Command-line Argument Handling
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a text classification model.")
    parser.add_argument("--model", type=str, choices=["distilbert", "minilm"], required=True, help="Model to use: 'distilbert' or 'minilm'")
    args = parser.parse_args()

    train_model(args.model)
