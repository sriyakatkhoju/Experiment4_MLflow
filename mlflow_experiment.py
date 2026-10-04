import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ---------------------------------------------------
# 1. Load the Loan Approval dataset
# ---------------------------------------------------

data = pd.read_csv("loan_approval.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ---------------------------------------------------
# 2. Separate input features and target variable
# ---------------------------------------------------

X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]


# ---------------------------------------------------
# 3. Convert categorical columns into numerical values
# ---------------------------------------------------

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


# Convert target Y/N into 1/0
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)


# ---------------------------------------------------
# 4. Split dataset into training and testing sets
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------------------------
# 5. Start MLflow experiment
# ---------------------------------------------------

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Loan Approval Prediction")


with mlflow.start_run() as run:

    print("\nMLflow Run started")
    print("Run ID:", run.info.run_id)


    # ---------------------------------------------------
    # 6. Train Random Forest Classifier
    # ---------------------------------------------------

    n_estimators = 100
    max_depth = 5
    random_state = 42

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    model.fit(X_train, y_train)


    # ---------------------------------------------------
    # 7. Log model parameters
    # ---------------------------------------------------

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)


    # ---------------------------------------------------
    # 8. Evaluate the model
    # ---------------------------------------------------

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


    # ---------------------------------------------------
    # 9. Log evaluation metric
    # ---------------------------------------------------

    mlflow.log_metric("accuracy", accuracy)


    # ---------------------------------------------------
    # 10. Create and save confusion matrix
    # ---------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Loan Approval Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.savefig("confusion_matrix.png")
    plt.close()


    # Log confusion matrix as artifact
    mlflow.log_artifact("confusion_matrix.png")


    # ---------------------------------------------------
    # 11. Save classification report
    # ---------------------------------------------------

    report = classification_report(y_test, y_pred, zero_division=0)

    with open("classification_report.txt", "w") as file:
        file.write(report)

    mlflow.log_artifact("classification_report.txt")


    # ---------------------------------------------------
    # 12. Log dataset as artifact
    # ---------------------------------------------------

    mlflow.log_artifact("loan_approval.csv")


    # ---------------------------------------------------
    # 13. Save trained model as MLflow artifact
    # ---------------------------------------------------
    mlflow.sklearn.log_model(
    sk_model=model,
    name="loan_approval_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
   )


    print("\nModel logged successfully!")
    print("Accuracy logged:", accuracy)
    print("MLflow Run ID:", run.info.run_id)

print("\nExperiment completed successfully!")