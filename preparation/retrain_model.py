import argparse
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ----------------------------
# 🛠️ Utility Functions
# ----------------------------

def load_dataset(file_path):
    """Loads the manually verified dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file '{file_path}' not found. Ensure it exists in the project directory.")

    df = pd.read_csv(file_path, delimiter="|")  # Ensure delimiter is correctly set
    df['label'] = df['label'].astype(int)  # Convert labels to integers
    
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
    """Loads the tokenizer and model from the specified fine-tuned model directory."""
    model_path = f"fine_tuned_{model_choice}"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory '{model_path}' not found. Ensure the model exists before retraining.")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    return tokenizer, model

# ----------------------------
# 🚀 Main Retraining Function
# ----------------------------

def retrain_model(model_choice, dataset_path="manual_dataset.csv"):
    """Retrains the specified fine-tuned model on manually verified data."""

    print(f"\n🚀 Retraining model: {model_choice} using dataset '{dataset_path}'\n")

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
        output_dir=f'./results_{model_choice}_v2',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,  # Retraining with 5 epochs
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir=f'./logs_{model_choice}_v2',
        logging_steps=10,
        metric_for_best_model="eval_accuracy",
        save_total_limit=2  # Limit saved checkpoints to avoid excessive disk usage
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
    print("\n🚀 Starting model retraining...\n")
    trainer.train()

    # Step 8: Save the retrained model and tokenizer
    save_path = f"fine_tuned_{model_choice}_v2"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n✅ Retrained model and tokenizer saved to '{save_path}'\n")

    # Step 9: Evaluate the model
    print("\n🔍 Evaluating on test dataset...\n")
    test_results = trainer.evaluate(test_dataset)

    # Display results
    print("\n📊 Test Dataset Evaluation:")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")

    # Step 10: Save results
    os.makedirs("evaluation", exist_ok=True)
    results_file = f"evaluation/{model_choice}_v2_performance.csv"
    pd.DataFrame([test_results]).to_csv(results_file, index=False)
    
    print(f"\n📁 Test results saved to: {results_file}\n")
    print("\n🎉 Retraining process completed successfully!")

# ----------------------------
# 🔹 Command-line Argument Handling
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain a fine-tuned text classification model on manually verified data.")
    parser.add_argument("--model", type=str, required=True, help="Fine-tuned model to retrain (e.g., 'distilbert' or 'minilm')")
    args = parser.parse_args()

    retrain_model(args.model)
