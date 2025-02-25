import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# Load dataset
df = pd.read_csv("labeled_dataset.csv")

# Ensure correct data types
df['label'] = df['label'].astype(int)

# Split dataset (80% train, 10% val, 10% test)
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

# Convert to Dataset object
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        [f"Title: {title} URL: {url} Sentence: {sentence}" for title, url, sentence in zip(examples['title'], examples['url'], examples['sentence'])],
        padding="max_length",
        truncation=True
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set dataset format
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=10,
    metric_for_best_model="eval_accuracy"
)

# Compute metrics function
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

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("fine_tuned_distilbert")
tokenizer.save_pretrained("fine_tuned_distilbert")

print("✅ Fine-tuned model and tokenizer saved to 'fine_tuned_distilbert'")

# Evaluate on test dataset
print("\n🔍 Evaluating on test dataset...")
test_results = trainer.evaluate(test_dataset)

# Display and save test results
print("\n📊 Test Dataset Evaluation:")
for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")

# Save results to CSV
os.makedirs("evaluation", exist_ok=True)
results_file = "evaluation/model_performance.csv"
pd.DataFrame([test_results]).to_csv(results_file, index=False)

print(f"✅ Test results saved to: {results_file}")
