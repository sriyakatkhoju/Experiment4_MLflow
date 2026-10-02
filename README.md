# Loan Approval Prediction using MLflow

## Experiment 4: MLflow Experiment Tracking

### Aim

To implement MLflow Experiment Tracking for recording model parameters, evaluation metrics, and model artifacts during the training of a Machine Learning model for Loan Approval Prediction.

## Algorithm

1. Import required libraries.
2. Load the Loan Approval dataset.
3. Perform basic preprocessing.
4. Split the dataset into training and testing sets.
5. Start an MLflow experiment.
6. Train a Random Forest Classifier.
7. Log model parameters.
8. Evaluate the model.
9. Log the evaluation metric.
10. Save the trained model as an MLflow artifact.
11. View the experiment in the MLflow UI.

## Model

Random Forest Classifier

### Parameters

- n_estimators = 100
- max_depth = 5
- random_state = 42

### Evaluation

- Accuracy = 0.75

### MLflow Experiment

Experiment Name:

`Loan Approval Prediction`

### Model Artifact

`loan_approval_model`

## How to Run

Create and activate the virtual environment:

```cmd
py -m venv .venv
.venv\Scripts\activate