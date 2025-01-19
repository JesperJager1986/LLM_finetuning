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
