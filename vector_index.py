import json
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load the datasets
dish_1_df = pd.read_csv("unlabeled_dataset.csv")
dish_2_df = pd.read_csv("labeled_dataset.csv")
dish_3_df = pd.read_csv("eval_labeled_dataset.csv")

# Combine unique dishes from all datasets into a single list
dishes = list(
    set(
        dish_1_df["dish"].unique().tolist()
        + dish_2_df["dish"].unique().tolist()
        + dish_3_df["dish"].unique().tolist()
    )
)
print(f"Unique dishes extracted: {len(dishes)}")

# Load a pretrained SentenceTransformer model
MODEL_PATH = "models/final"  # Replace with the path to your trained model if necessary
model = SentenceTransformer(MODEL_PATH)

# Generate embeddings for each dish
dish_embeddings = {dish: model.encode(dish).tolist() for dish in dishes}

# Save the embeddings to a JSON file
output_path = "dish_vector_index.json"
with open(output_path, "w") as f:
    json.dump(dish_embeddings, f)

print(f"Vector index for dishes saved to: {output_path}")
