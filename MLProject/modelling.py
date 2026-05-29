import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

# =====================================
# SET EXPERIMENT
# =====================================

mlflow.set_experiment(
    "Employee_Attrition_Experiment"
)

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv(
    "Employee-Attrition-Dataset_preprocessing/employee_attrition_processed.csv"
)

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# =====================================
# SPLIT DATA
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =====================================
# TRAINING
# =====================================

with mlflow.start_run():

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(f"Accuracy: {accuracy}")

    # Log parameter
    mlflow.log_param(
        "random_state",
        42
    )

    # Log metric
    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    # Signature
    signature = infer_signature(
        X_train,
        model.predict(X_train)
    )

    # Save model ke MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5)
    )
