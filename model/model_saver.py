from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.config import Config


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
