from config.config import Config
from pipeline.pipeline_finetuning import FineTuningPipeline



if __name__ == '__main__':
    config = Config()
    pipeline = FineTuningPipeline(config)
    pipeline.run()
