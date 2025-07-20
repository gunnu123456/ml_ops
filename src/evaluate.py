"""
Model evaluation script for MLOps pipeline.
"""

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os


def load_data_and_model():
    """Load the trained model and test data."""
    # Load data from parquet
    df = pd.read_parquet("data/iris.parquet")
    
    # Encode the target column (same as in train.py)
    from sklearn.preprocessing import OrdinalEncoder
    encoder = OrdinalEncoder()
    df["target"] = encoder.fit_transform(df[["species"]]).astype(int)
    
    X = df.drop(columns=["species", "target"])
    y = df["target"]
    
    # Split data (same as in training)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Load model
    model = joblib.load("models/model.joblib")
    
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and generate metrics."""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save evaluation report
    with open("report.md", "a") as f:
        f.write(f"\n# Model Evaluation Report\n\n")
        f.write(f"**Accuracy:** {accuracy:.4f}\n\n")
        f.write(f"## Classification Report\n```\n{report}\n```\n\n")
        f.write(f"## Confusion Matrix\n```\n{cm}\n```\n")
    
    return accuracy, report, cm


if __name__ == "__main__":
    print("Starting model evaluation...")
    model, X_test, y_test = load_data_and_model()
    evaluate_model(model, X_test, y_test)
    print("Model evaluation completed!")
