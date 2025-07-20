import subprocess
import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder


MODEL_PATH = "models/test_model.joblib"
DATA_PATH = "data/iris.parquet"


def test_train_runs_successfully():
    """Ensure train.py runs without crashing."""
    result = subprocess.run(
        ["python", "src/train.py", "--data", DATA_PATH, "--model_out", MODEL_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    assert result.returncode == 0, f"Train script failed: {result.stderr}"


def test_model_file_created():
    """Check that model output file was created."""
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"


def test_model_accuracy_threshold():
    """Check that the trained model achieves minimum accuracy (e.g. 0.90)."""
    df = pd.read_parquet(DATA_PATH)
    df["target"] = OrdinalEncoder().fit_transform(df[["species"]]).astype(int)
    
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    assert acc > 0.90, f"Accuracy below threshold: {acc:.4f}"


def test_confusion_matrix():
    """Generate and print confusion matrix for manual inspection."""
    df = pd.read_parquet(DATA_PATH)
    df["target"] = OrdinalEncoder().fit_transform(df[["species"]]).astype(int)

    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    print("\n[CONFUSION MATRIX]")
    print(cm)

    # You can optionally assert the matrix shape
    assert cm.shape == (3, 3), "Expected a 3x3 confusion matrix"

def teardown_module(module):
    """Cleanup test artifacts after tests run."""
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
