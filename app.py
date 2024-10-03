import mlflow

# Load model 1 from the Databricks Model Registry
model_1_uri = "models:/llm_model_1/production"  # Adjust with the actual model name and stage/version
llm_model_1 = mlflow.pyfunc.load_model(model_1_uri)

# Load model 2 from the Databricks Model Registry
model_2_uri = "models:/llm_model_2/production"  # Adjust with the actual model name and stage/version
llm_model_2 = mlflow.pyfunc.load_model(model_2_uri)

# List of models to evaluate
models = [llm_model_1, llm_model_2]

# Function to generate answers using the loaded models
def generate_answer(model, question):
    # This assumes the model is callable (like a function)
    # and can process the question directly.
    prediction = model.predict(pd.DataFrame([question], columns=["question"]))
    return prediction[0]  # Assuming the output is a list or array with answers

# Example question
question = "What is the capital of France?"

# Generate answers from each model
answer_model_1 = generate_answer(llm_model_1, question)
answer_model_2 = generate_answer(llm_model_2, question)

print(f"Model 1 Answer: {answer_model_1}")
print(f"Model 2 Answer: {answer_model_2}")


