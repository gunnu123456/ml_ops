import subprocess
import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from feast import FeatureStore
from datetime import datetime
import unittest # Import the unittest framework
import sys

# Define paths for data and model artifacts
MODEL_PATH = "models/test_model.joblib"
DATA_PATH = "data/iris.parquet"
CONFUSION_MATRIX_FILE = "confusion_matrix.txt"

# Ensure Feast directory is in sys.path for importing feature definitions
sys.path.append(os.path.join(os.getcwd(), "feast"))
try:
    from iris_features import iris_view, flower # Assuming 'flower' is your entity
except ImportError as e:
    print(f"Error importing Feast features in test: {e}. Make sure feast/iris_features.py exists and defines 'iris_view' and 'flower'.")
    sys.exit(1)

# Helper function to prepare data using Feast, mirroring src/train.py
def prepare_data_with_feast(data_path):
    """
    Loads data, prepares entity_df, fetches features from Feast,
    and returns X, y for model evaluation.
    """
    df_original = pd.read_parquet(data_path)

    # Add entity_id and timestamp for Feast
    df_original["flower_id"] = df_original.index.astype(int)
    df_original["event_timestamp"] = datetime.utcnow()

    # Initialize FeatureStore
    feast_repo_path = os.path.abspath("feast")
    store = FeatureStore(repo_path=feast_repo_path)

    # Define entity dataframe for historical feature retrieval
    entity_df = df_original[["flower_id", "event_timestamp"]]

    # Specify features to fetch from Feast
    features_to_fetch = [
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width",
    ]

    # Retrieve offline features
    features_df = store.get_historical_features(
        entity_df=entity_df,
        features=features_to_fetch
    ).to_df()

    # Encode labels
    encoder = OrdinalEncoder()
    df_original["target"] = encoder.fit_transform(df_original[["species"]]).astype(int)

    # Merge features and target based on join keys
    full_df = pd.merge(
        features_df,
        df_original[["flower_id", "event_timestamp", "target"]],
        on=["flower_id", "event_timestamp"],
        how="inner"
    )

    # Prepare X and y for training/evaluation
    X = full_df.drop(columns=["flower_id", "event_timestamp", "target"])
    y = full_df["target"]

    return X, y

# Define a test class that inherits from unittest.TestCase
class TestTrainScript(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up for all tests in this class.
        This runs once before any test methods are run.
        We'll run the train script here to ensure the model is created for subsequent tests.
        """
        print("\n--- Running setUpClass: Executing src/train.py ---")
        result = subprocess.run(
            ["python", "src/train.py", "--data", DATA_PATH, "--model_out", MODEL_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        print("--- stdout from train.py ---")
        print(result.stdout)
        print("--- stderr from train.py ---")
        print(result.stderr)
        cls.assertEqual(result.returncode, 0, f"Train script failed during setup: {result.stderr}")
        cls.assertTrue(os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH} after train script ran during setup.")
        print(f"Model file created successfully at {MODEL_PATH} during setup.")

        # Prepare data once for all tests in this class
        cls.X, cls.y = prepare_data_with_feast(DATA_PATH)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, stratify=cls.y, random_state=42
        )
        cls.model = joblib.load(MODEL_PATH)


    def test_model_file_exists(self):
        """Check that model output file was created."""
        self.assertTrue(os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}")

    def test_model_accuracy_threshold(self):
        """Check that the trained model achieves minimum accuracy (e.90)."""
        preds = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)

        print(f"Model accuracy: {acc:.4f}")
        self.assertGreater(acc, 0.90, f"Accuracy below threshold: {acc:.4f}")

    def test_confusion_matrix(self):
        """Generate and save confusion matrix for inspection."""
        preds = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, preds)

        print("\n[CONFUSION MATRIX]")
        print(cm)

        # Save confusion matrix to a file
        with open(CONFUSION_MATRIX_FILE, "w") as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
            f.write("\n")
        print(f"Confusion matrix saved to {CONFUSION_MATRIX_FILE}")

        # Assert the matrix shape
        self.assertEqual(cm.shape, (3, 3), "Expected a 3x3 confusion matrix")

    @classmethod
    def tearDownClass(cls):
        """
        Clean up test artifacts after all tests in this class run.
        This runs once after all test methods are run.
        """
        print("--- Running tearDownClass: Cleaning up artifacts ---")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            print(f"Removed {MODEL_PATH}")
        if os.path.exists(CONFUSION_MATRIX_FILE):
            os.remove(CONFUSION_MATRIX_FILE)
            print(f"Removed {CONFUSION_MATRIX_FILE}")

# This allows running the tests directly from the command line
if __name__ == '__main__':
    unittest.main()

