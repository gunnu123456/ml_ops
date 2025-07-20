import pandas as pd
from datetime import datetime
import argparse

# ----------------- ARGUMENT PARSING -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--in_file", default="data/iris.csv")
parser.add_argument("--out_file", default="data/iris_feast.parquet")
args = parser.parse_args()

# ----------------- LOAD & ENRICH DATA -----------------
df = pd.read_csv(args.in_file)

df["flower_id"] = df.index.astype(int)
df["event_timestamp"] = datetime.utcnow()

# ----------------- SAVE PARQUET FOR FEAST -----------------
df.to_parquet(args.out_file, index=False)

# ----------------- SAVE PARQUET FOR TRAINING -----------------
df[["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]].to_parquet("data/iris.parquet", index=False)

print("✅ Saved both iris_feast.parquet and iris.parquet")
