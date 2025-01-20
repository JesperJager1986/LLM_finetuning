from datasets import load_dataset

from config.config import Config
import pandas as pd

class DatasetLoader:
    def __init__(self, config: Config):
        self.config = config
        self.dataset = self.load_dataset(config.dataset_name)

    @staticmethod
    def load_dataset(name: str):
        print("Loading dataset...")
        return load_dataset(name)

    def load_and_split(self, name: str):
        size = 1 - self.config.train_size if name == "train" else self.config.test_size
        reduced_dataset = self.dataset[name].train_test_split(test_size=size)[name]
        self._describe_distribution(pd.Series(reduced_dataset["label"]))
        return reduced_dataset

    @staticmethod
    def _describe_distribution(labels: pd.Series) -> None:
        label_counts = labels.value_counts(normalize=True) * 100  # normalize=True gives percentages
        print(label_counts)


