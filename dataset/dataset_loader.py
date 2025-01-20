from datasets import load_dataset

from config.config import Config
import pandas as pd

class DatasetLoader:
    def __init__(self, config: Config):
        self.config = config

    #todo look into this function converted into pandas dataframe
    def load_and_split(self):
        print("Loading dataset...")
        dataset = load_dataset(self.config.dataset_name)
        train_dataset = dataset["train"].train_test_split(test_size=1-self.config.train_size)["train"]
        test_dataset = dataset["train"].train_test_split(test_size=self.config.test_size)["test"]

        self._describe_distribution(pd.Series(train_dataset["label"]))
        self._describe_distribution(pd.Series(train_dataset["label"]))
        return train_dataset, test_dataset

    @staticmethod
    def _describe_distribution(labels: pd.Series) -> None:
        label_counts = labels.value_counts(normalize=True) * 100  # normalize=True gives percentages
        print(label_counts)


