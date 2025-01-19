from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config.config import Config


class ModelHandler:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name, num_labels=self.config.num_labels)

    def tokenize(self, dataset):
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding=self.config.padding, truncation=self.config.truncation)
        return dataset.map(tokenize_function, batched=self.config.map_batched)
