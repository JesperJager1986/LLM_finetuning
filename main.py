from config.config import Config
from pipeline.pipeline_finetuning import FineTuningPipeline
import mlflow





if __name__ == '__main__':
    # Set the tracking URI to the MLflow server
    mlflow.set_tracking_uri("http://0.0.0.0:5000")

    # Create or set an experiment
    experiment_name = "Sample Experiment"
    mlflow.set_experiment(experiment_name)

    try:
        # Verify MLflow Tracking URI
        tracking_uri = mlflow.get_tracking_uri()
        print(f"MLflow Tracking URI: {tracking_uri}")

        # Verify connection to the backend store
        mlflow.list_experiments()
        print("✅ MLflow setup is correct. Backend store is accessible.")

    except Exception as e:
        print("❌ MLflow setup error:")
        print(str(e))
        exit(1)




    mlflow.autolog()
    mlflow.set_experiment('fine tuning llm')
    with mlflow.start_run():
        config = Config()
        pipeline = FineTuningPipeline(config)
        pipeline.run()
