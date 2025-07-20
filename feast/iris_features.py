from feast import Entity, FeatureView, Field, FileSource, FeatureStore
from feast.types import Float32, Int64
from datetime import timedelta
import os

# Define source
iris_source = FileSource(
    path="../data/iris_feast.parquet",
    timestamp_field="event_timestamp",
)

# Define entity
flower = Entity(name="flower_id", join_keys=["flower_id"])

# Define feature view
iris_view = FeatureView(
    name="iris_features",
    entities=[flower],
    ttl=timedelta(days=1),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
    ],
    source=iris_source,
    online=True,
)

# ✅ This is only needed if you call apply() in this file
if __name__ == "__main__":
    print("[INFO] Initializing FeatureStore with correct path feast/")
    store = FeatureStore(repo_path="feast")  # <-- Fixed path
    store.apply([flower, iris_view])
