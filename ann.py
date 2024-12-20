import faiss
import numpy as np
import json

from sentence_transformers import SentenceTransformer

# Load the vector index from the JSON file
vector_index_path = "dish_vector_index.json"
with open(vector_index_path, "r") as f:
    dish_vectors = json.load(f)

# Prepare data for ANN index
dish_names = list(dish_vectors.keys())
vectors = np.array(list(dish_vectors.values())).astype("float32")

# Build the FAISS index
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance
index.add(vectors)  # Add vectors to the index
print(f"FAISS index built with {index.ntotal} items.")

# Save the FAISS index for later use
faiss_index_path = "dish_ann_index.faiss"
faiss.write_index(index, faiss_index_path)
print(f"FAISS index saved to {faiss_index_path}")


# Function to query the ANN index
def query_ann_index(query: str, model, top_k: int = 5):
    """
    Query the ANN index with a dish query and return the top_k closest dishes.

    Parameters:
    ----------
    query : str
        The dish query string.
    model : SentenceTransformer
        The embedding model to vectorize the query.
    top_k : int, optional
        Number of closest matches to return (default is 5).

    Returns:
    -------
    list of tuples
        A list of (dish_name, distance) pairs sorted by similarity.
    """
    # Generate query vector
    query_vector = np.array([model.encode(query)]).astype("float32")

    # Perform the search
    distances, indices = index.search(query_vector, top_k)

    # Map indices to dish names
    results = [(dish_names[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results


# Example usage
if __name__ == "__main__":
    # Load the embedding model
    MODEL_PATH = "models/final"  # Replace with your model path
    model = SentenceTransformer(MODEL_PATH)

    # Query the ANN index
    query_dish = "vegan burger"
    top_matches = query_ann_index(query_dish, model, top_k=5)

    print(f"Top matches for '{query_dish}':")
    for match, distance in top_matches:
        print(f"Dish: {match}, Distance: {distance}")
