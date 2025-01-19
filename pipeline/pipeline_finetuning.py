from config.config import Config
from dataset.dataset_loader import DatasetLoader
from model.model_handler import ModelHandler
from model.model_saver import ModelSaver
from model.model_trainer import TrainerHandler


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
