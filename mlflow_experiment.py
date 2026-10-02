import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 1. Load the Loan Approval dataset
data = pd.read_csv("loan_approval.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# 2. Separate input features and target variable
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]


# 3. Convert categorical columns into numerical values
categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed"
]

for column in categorical_columns:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column])


# Convert target variable Y/N into 1/0
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)


# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 5. Set MLflow experiment
mlflow.set_experiment("Loan Approval Prediction")


# 6. Start an MLflow run
with mlflow.start_run() as run:

    # 7. Random Forest parameters
    n_estimators = 100
    max_depth = 5
    random_state = 42

    # 8. Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    # 9. Train the model
    model.fit(X_train, y_train)

    # 10. Make predictions
    y_pred = model.predict(X_test)

    # 11. Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # 12. Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    # 13. Log accuracy
    mlflow.log_metric("accuracy", accuracy)

    # 14. Log trained model
    mlflow.sklearn.log_model(
        model,
        name="loan_approval_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    # 15. Display experiment information
    print()
    print("========== MLflow Experiment ==========")
    print("Experiment: Loan Approval Prediction")
    print("Run ID:", run.info.run_id)
    print("n_estimators:", n_estimators)
    print("max_depth:", max_depth)
    print("random_state:", random_state)
    print("Accuracy:", accuracy)
    print("Model artifact: loan_approval_model")
    print("=======================================")