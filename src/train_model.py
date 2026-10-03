import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =========================================================
# 1. PROJECT PATHS
# =========================================================

DATA_PATH = Path("data/german_credit.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# =========================================================
# 2. MLFLOW SETUP
# =========================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Credit_Risk_Prediction")


# =========================================================
# 3. LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("CREDIT RISK MODEL TRAINING + MLFLOW")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")


# =========================================================
# 4. FEATURES AND TARGET
# =========================================================

X = df.drop("credit_risk", axis=1)
y = df["credit_risk"]


# =========================================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# =========================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nData split:")
print(f"Training   : {len(X_train)} rows")
print(f"Validation : {len(X_val)} rows")
print(f"Testing    : {len(X_test)} rows")


# =========================================================
# 6. IDENTIFY COLUMN TYPES
# =========================================================

categorical_columns = X.select_dtypes(
    include=["str"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["str"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# =========================================================
# 7. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)


# =========================================================
# 8. DEFINE MODELS
# =========================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=5000,
        solver="liblinear",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
}


# =========================================================
# 9. TRAIN MODELS + MLFLOW TRACKING
# =========================================================

results = []
trained_models = {}

for model_name, model in models.items():

    print("\n" + "-" * 60)
    print(f"Training: {model_name}")
    print("-" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Start MLflow run
    with mlflow.start_run(run_name=model_name):

        # Train model
        pipeline.fit(X_train, y_train)

        # Validation predictions
        y_val_pred = pipeline.predict(X_val)
        y_val_prob = pipeline.predict_proba(X_val)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(
            y_val,
            y_val_pred
        )

        precision = precision_score(
            y_val,
            y_val_pred,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            y_val_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            y_val_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_val,
            y_val_prob
        )

        # Display metrics
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")

        # Log parameters
        mlflow.log_param(
            "model_name",
            model_name
        )

        # Log validation metrics
        mlflow.log_metric(
            "val_accuracy",
            accuracy
        )

        mlflow.log_metric(
            "val_precision",
            precision
        )

        mlflow.log_metric(
            "val_recall",
            recall
        )

        mlflow.log_metric(
            "val_f1",
            f1
        )

        mlflow.log_metric(
            "val_roc_auc",
            roc_auc
        )

        # Log model using cloudpickle.
        # This avoids the skops trusted-tree error.
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format="cloudpickle"
        )

        # Save run ID for display
        print(
            f"MLflow Run ID: "
            f"{mlflow.active_run().info.run_id}"
        )

    # Store result
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    # Store trained model
    trained_models[model_name] = pipeline


# =========================================================
# 10. MODEL COMPARISON
# =========================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# =========================================================
# 11. SELECT BEST MODEL
# =========================================================

best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]

print("\n" + "=" * 60)
print(f"Selected model: {best_model_name}")
print("=" * 60)


# =========================================================
# 12. FINAL TEST EVALUATION
# =========================================================

y_test_pred = best_model.predict(X_test)
y_test_prob = best_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

test_precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

test_roc_auc = roc_auc_score(
    y_test,
    y_test_prob
)


print("\nFINAL TEST RESULTS")
print("-" * 40)

print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1 Score : {test_f1:.4f}")
print(f"ROC-AUC  : {test_roc_auc:.4f}")


# =========================================================
# 13. SAVE ALL MODELS
# =========================================================

for model_name, trained_model in trained_models.items():

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    joblib.dump(
        trained_model,
        MODEL_DIR / filename
    )


# Save best model separately
joblib.dump(
    best_model,
    MODEL_DIR / "best_model.joblib"
)


# =========================================================
# 14. SAVE RESULTS CSV
# =========================================================

results_df.to_csv(
    REPORT_DIR / "model_results.csv",
    index=False
)


# =========================================================
# 15. MLFLOW FINAL TEST RUN
# =========================================================

with mlflow.start_run(
    run_name=f"{best_model_name}_Final_Test"
):

    mlflow.log_param(
        "selected_model",
        best_model_name
    )

    mlflow.log_metric(
        "test_accuracy",
        test_accuracy
    )

    mlflow.log_metric(
        "test_precision",
        test_precision
    )

    mlflow.log_metric(
        "test_recall",
        test_recall
    )

    mlflow.log_metric(
        "test_f1",
        test_f1
    )

    mlflow.log_metric(
        "test_roc_auc",
        test_roc_auc
    )

    # Log final model
    mlflow.sklearn.log_model(
        best_model,
        name="best_model",
        serialization_format="cloudpickle"
    )

    print(
        f"\nFinal MLflow Run ID: "
        f"{mlflow.active_run().info.run_id}"
    )


# =========================================================
# 16. COMPLETION MESSAGE
# =========================================================

print("\n")
print("=" * 60)
print("PROJECT TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nModels saved inside:")
print("models/")

print("\nResults saved to:")
print("reports/model_results.csv")

print("\nMLflow database:")
print("mlflow.db")

print("\nMLflow Experiment:")
print("Credit_Risk_Prediction")

print("\nMachine Learning + MLflow completed successfully!")