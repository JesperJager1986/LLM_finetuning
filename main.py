# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
#import torch

# Step 1: Load the dataset
print("Loading dataset...")
dataset = load_dataset("imdb")

# Split the dataset into training and validation sets
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Step 2: Load the pre-trained model and tokenizer
model_name = "distilbert-base-uncased"  # A small Hugging Face model
print(f"Loading model and tokenizer: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification

# Step 3: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

print("Tokenizing dataset...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove unused columns for training
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# Step 4: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save model checkpoints and logs
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    learning_rate=2e-5,  # Learning rate
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    num_train_epochs=3,  # Number of training epochs
    weight_decay=0.01,  # Weight decay for optimization
    logging_dir="./logs",  # Directory to save logs
    logging_steps=10,  # Log every 10 steps
    save_strategy="epoch",  # Save the model at each epoch
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="accuracy",  # Choose the best model based on accuracy
)

# Step 5: Define a function to compute metrics
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Step 6: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Step 7: Fine-tune the model
print("Starting fine-tuning...")
trainer.train()

# Step 8: Save the fine-tuned model
output_dir = "./fine_tuned_distilbert"
print(f"Saving the fine-tuned model to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Fine-tuning complete and model saved!")
