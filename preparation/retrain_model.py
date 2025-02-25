import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# ----------------------------
# 🔍 Step 1: Load Manual Dataset
# ----------------------------
dataset_path = "manual_dataset.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file '{dataset_path}' not found. Please ensure it exists in the project directory.")

# Ensure delimiter is properly recognized
df = pd.read_csv(dataset_path, delimiter="|")


# Ensure label column is integer
df['label'] = df['label'].astype(int)

# ----------------------------
# 🔍 Step 2: Split Dataset
# ----------------------------
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

# Convert to Dataset object
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# ----------------------------
# 🛠️ Step 3: Load Tokenizer & Model (From Existing Fine-Tuned Model)
# ----------------------------
tokenizer = DistilBertTokenizer.from_pretrained("fine_tuned_distilbert")
model = DistilBertForSequenceClassification.from_pretrained("fine_tuned_distilbert")

# ----------------------------
# 🔍 Step 4: Tokenize Dataset
# ----------------------------
def tokenize_function(examples):
    return tokenizer(
        [f"Title: {title} URL: {url} Sentence: {sentence}" for title, url, sentence in zip(examples['title'], examples['url'], examples['sentence'])],
        padding="max_length",
        truncation=True,
        max_length=512
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set dataset format
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# ----------------------------
# ⚙️ Step 5: Set Training Arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir='./results_v3',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,  # Increase to 5 for more training
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir='./logs_v3',
    logging_steps=10,
    metric_for_best_model="eval_accuracy",
    save_total_limit=2  # Limit saved checkpoints to avoid disk usage
)

# ----------------------------
# 📊 Step 6: Define Compute Metrics
# ----------------------------
def compute_metrics(pred):
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

# ----------------------------
# 🚀 Step 7: Initialize Trainer & Train Model
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("\n🚀 Starting model retraining with manual dataset...\n")
trainer.train()

# ----------------------------
# 💾 Step 8: Save Retrained Model
# ----------------------------
model.save_pretrained("fine_tuned_distilbert_v2")
tokenizer.save_pretrained("fine_tuned_distilbert_v2")

print("\n✅ Retrained model and tokenizer saved to 'fine_tuned_distilbert_v2'")

# ----------------------------
# 🔍 Step 9: Evaluate on Test Dataset
# ----------------------------
print("\n🔍 Evaluating on test dataset...")
test_results = trainer.evaluate(test_dataset)

# Display test results
print("\n📊 Test Dataset Evaluation (Manual Dataset):")
for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")

# ----------------------------
# 💾 Step 10: Save Test Results
# ----------------------------
os.makedirs("evaluation", exist_ok=True)
results_file = "evaluation/manual_dataset_performance.csv"
pd.DataFrame([test_results]).to_csv(results_file, index=False)

print(f"\n📁 Test results saved to: {results_file}")

print("\n🎉 Retraining process completed successfully!")
