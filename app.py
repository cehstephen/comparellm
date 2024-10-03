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


import mlflow
from sklearn.metrics import accuracy_score  # Or any other metrics

# List of questions and ground truth answers
questions = [
    "What is the capital of France?",
    "Who wrote '1984'?",
    "What is the tallest mountain?"
]

ground_truth = ["Paris", "George Orwell", "Mount Everest"]

# Iterate over each model and evaluate
model_names = ["llm_model_1", "llm_model_2"]
for model, model_name in zip(models, model_names):
    
    # Generate predictions
    predictions = [generate_answer(model, question) for question in questions]
    
    # Evaluate accuracy (exact match)
    accuracy = accuracy_score(ground_truth, predictions)
    
    # Log results in MLflow
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log each question, prediction, and ground truth
        for q, pred, truth in zip(questions, predictions, ground_truth):
            mlflow.log_param(f"question: {q}", f"prediction: {pred}, ground_truth: {truth}")
        
        print(f"Model: {model_name}, Accuracy: {accuracy}")



