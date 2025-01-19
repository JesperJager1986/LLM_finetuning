from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from sklearn.metrics import accuracy_score, f1_score

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


class DatasetLoader:
    def __init__(self, config: Config):
        self.config = config

    def load_and_split(self):
        print("Loading dataset...")
        dataset = load_dataset(self.config.dataset_name)
        train_dataset = dataset["train"].train_test_split(test_size=1-self.config.train_size)["train"]
        test_dataset = dataset["train"].train_test_split(test_size=self.config.test_size)["test"]
        return train_dataset, test_dataset


class ModelHandler:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name, num_labels=self.config.num_labels)

    def tokenize(self, dataset):
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding=self.config.padding, truncation=self.config.truncation)
        return dataset.map(tokenize_function, batched=self.config.map_batched)


class TrainerHandler:
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, config: Config, train_dataset, test_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), axis=1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        return {"accuracy": acc, "f1": f1}

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            evaluation_strategy=self.config.evaluation_strategy,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            num_train_epochs=self.config.num_train_epochs,
            weight_decay=self.config.weight_decay,
            logging_dir=self.config.logging_dir,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        print("Starting fine-tuning...")
        trainer.train()


class ModelSaver:
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def save(self):
        print(f"Saving the fine-tuned model to {self.config.fine_tuned_model_dir}...")
        self.model.save_pretrained(self.config.fine_tuned_model_dir)
        self.tokenizer.save_pretrained(self.config.fine_tuned_model_dir)
        print("Fine-tuning complete and model saved!")


class FineTuningPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.dataset_loader = DatasetLoader(config)
        self.model_handler = ModelHandler(config)
        self.train_dataset, self.test_dataset = self.dataset_loader.load_and_split()
        self.model_handler.tokenize(self.train_dataset)
        self.model_handler.tokenize(self.test_dataset)
        self.trainer_handler = TrainerHandler(self.model_handler.model, self.model_handler.tokenizer, self.config, self.train_dataset, self.test_dataset)
        self.model_saver = ModelSaver(self.model_handler.model, self.model_handler.tokenizer, self.config)

    def run(self):
        self.trainer_handler.train()
        self.model_saver.save()


# Main execution
config = Config()
pipeline = FineTuningPipeline(config)
pipeline.run()
