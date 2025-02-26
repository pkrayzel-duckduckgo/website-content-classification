import argparse
import os
import torch
import pandas as pd
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from utils import load_dataset, split_dataset, tokenize_function, compute_metrics, get_model_and_tokenizer

# ----------------------------
# 🚀 Main Training Function
# ----------------------------

def train_model(model_choice, dataset_path, resume, delimiter):
    """Trains or retrains a model based on user choice."""

    print(f"\n🚀 {'Resuming training' if resume else 'Training from scratch'} with model: {model_choice}\n")

    # Step 1: Load and split dataset
    df = load_dataset(dataset_path, delimiter)
    train_df, val_df, test_df = split_dataset(df)

    # Step 2: Load tokenizer and model
    tokenizer, model = get_model_and_tokenizer(model_choice, resume)

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
        output_dir=f'./results_{model_choice}{"_v2" if resume else ""}',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5 if resume else 3,  # Retraining gets more epochs
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir=f'./logs_{model_choice}{"_v2" if resume else ""}',
        logging_steps=10,
        metric_for_best_model="eval_accuracy",
        save_total_limit=2
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
    print("\n🚀 Starting training...\n")
    trainer.train()

    # Step 8: Save the model and tokenizer
    save_path = f"fine_tuned_{model_choice}{'_v2' if resume else ''}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n✅ Model and tokenizer saved to '{save_path}'\n")

    # Step 9: Evaluate the model
    print("\n🔍 Evaluating on test dataset...\n")
    test_results = trainer.evaluate(test_dataset)

    # Display and save results
    os.makedirs("evaluation", exist_ok=True)
    results_file = f"evaluation/{model_choice}{'_v2' if resume else ''}_performance.csv"
    pd.DataFrame([test_results]).to_csv(results_file, index=False)
    
    print(f"\n📁 Test results saved to: {results_file}\n")
    print("\n🎉 Training process completed successfully!")

# ----------------------------
# 🔹 Command-line Argument Handling
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or retrain a text classification model.")
    parser.add_argument("--model", type=str, choices=["distilbert", "minilm"], required=True, help="Model to use: 'distilbert' or 'minilm'")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset file.")
    parser.add_argument("--resume", action="store_true", help="Resume training from fine-tuned model.")
    parser.add_argument("--delimiter", type=str, default=",", help="CSV delimiter (default: ','). Use '|' for pipe-separated files.")
    args = parser.parse_args()

    train_model(args.model, args.dataset, args.resume, args.delimiter)
