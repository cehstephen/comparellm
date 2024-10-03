import mlflow

# Load model 1 from the Databricks Model Registry
model_1_uri = "models:/llm_model_1/production"  # Adjust with the actual model name and stage/version
llm_model_1 = mlflow.pyfunc.load_model(model_1_uri)

# Load model 2 from the Databricks Model Registry
model_2_uri = "models:/llm_model_2/production"  # Adjust with the actual model name and stage/version
llm_model_2 = mlflow.pyfunc.load_model(model_2_uri)

# List of models to evaluate
models = [llm_model_1, llm_model_2]

