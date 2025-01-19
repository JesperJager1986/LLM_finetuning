# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

class Config:
    def __init__(
        self,
        dataset_name: str = "imdb",
        train_size: float = 0.03,
        test_size: float = 0.02,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 2,
        padding: str = "max_length",
        truncation: bool = True,
        map_batched: bool = True,
        output_dir: str = "./results",
        evaluation_strategy: str = "epoch",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        logging_dir: str = "./logs",
        logging_steps: int = 10,
        save_strategy: str = "epoch",
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "accuracy",
        fine_tuned_model_dir: str = "./fine_tuned_distilbert",
        per_device_train_batch_size: int = 2,
        per_device_eval_batch_size: int = 2,
        num_train_epochs: int = 5,
    ):
        self.dataset_name = dataset_name
        self.train_size = train_size
        self.test_size = test_size
        self.model_name = model_name
        self.num_labels = num_labels
        self.padding = padding
        self.truncation = truncation
        self.map_batched = map_batched
        self.output_dir = output_dir
        self.evaluation_strategy = evaluation_strategy
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps
        self.save_strategy = save_strategy
        self.load_best_model_at_end = load_best_model_at_end
        self.metric_for_best_model = metric_for_best_model
        self.fine_tuned_model_dir = fine_tuned_model_dir
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.num_train_epochs = num_train_epochs

config = Config()
# Step 1: Load the dataset
print("Loading dataset...")
dataset = load_dataset(config.dataset_name)

# Split the dataset into training and validation sets
#train_dataset = dataset["train"]
train_dataset = dataset["train"].train_test_split(test_size=1-config.train_size)["train"]
test_dataset = dataset["train"].train_test_split(test_size=config.test_size)["test"]

# Step 2: Load the pre-trained model and tokenizer
#model_name = "gaunernst/bert-tiny-uncased"
#model_name = "distilbert-base-uncased"  # A small Hugging Face model
print(f"Loading model and tokenizer: {config.model_name}...")
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels)  # Binary classification

# Step 3: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding=config.padding, truncation=config.truncation)

print("Tokenizing dataset...")
train_dataset = train_dataset.map(tokenize_function, batched=config.map_batched)
test_dataset = test_dataset.map(tokenize_function, batched=config.map_batched)

# Remove unused columns for training
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# Step 4: Define training arguments
training_args = TrainingArguments(
    output_dir= config.output_dir ,  # Directory to save model checkpoints and logs
    evaluation_strategy=config.evaluation_strategy,  # Evaluate at the end of each epoch
    learning_rate=config.learning_rate,  # Learning rate
    per_device_train_batch_size=config.per_device_train_batch_size,  # Batch size for training
    per_device_eval_batch_size=config.per_device_eval_batch_size,  # Batch size for evaluation
    num_train_epochs=config.num_train_epochs,  # Number of training epochs
    weight_decay=config.weight_decay,  # Weight decay for optimization
    logging_dir=config.logging_dir,  # Directory to save logs
    logging_steps=config.logging_steps,  # Log every 10 steps
    save_strategy=config.save_strategy,  # Save the model at each epoch
    load_best_model_at_end=config.load_best_model_at_end,  # Load the best model at the end of training
    metric_for_best_model=config.metric_for_best_model,  # Choose the best model based on accuracy
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
