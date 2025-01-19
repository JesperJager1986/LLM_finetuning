from datasets import load_dataset

from config.config import Config


class DatasetLoader:
    def __init__(self, config: Config):
        self.config = config

    def load_and_split(self):
        print("Loading dataset...")
        dataset = load_dataset(self.config.dataset_name)
        train_dataset = dataset["train"].train_test_split(test_size=1-self.config.train_size)["train"]
        test_dataset = dataset["train"].train_test_split(test_size=self.config.test_size)["test"]
        return train_dataset, test_dataset


