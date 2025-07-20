"""
Generate metrics visualization for MLOps pipeline.
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np


def load_data_and_model():
    """Load the trained model and test data."""
    # Load data from parquet
    df = pd.read_parquet("data/iris.parquet")

    # Encode target (same logic as train.py)
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


def plot_metrics(model, X_test, y_test):
    """Generate and save metrics visualization."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Plot feature importance (works only for tree-based models)
    if hasattr(model, "feature_importances_"):
        feature_names = X_test.columns
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]

        axes[1].bar(range(len(importance)), importance[indices])
        axes[1].set_title('Feature Importance')
        axes[1].set_xlabel('Feature')
        axes[1].set_ylabel('Importance')
        axes[1].set_xticks(range(len(importance)))
        axes[1].set_xticklabels([feature_names[i] for i in indices], rotation=45)
    else:
        axes[1].text(0.5, 0.5, 'Feature importance not available',
                     horizontalalignment='center', verticalalignment='center')
        axes[1].set_title('Feature Importance')

    plt.tight_layout()
    plt.savefig("metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ Metrics plot saved as 'metrics.png'")


if __name__ == "__main__":
    print("Generating metrics visualization...")
    model, X_test, y_test = load_data_and_model()
    plot_metrics(model, X_test, y_test)
    print("Metrics visualization completed!")
