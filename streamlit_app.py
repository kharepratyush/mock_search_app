import streamlit as st
import faiss
import numpy as np
import requests
import json

# Load the FAISS index and metadata
FAISS_INDEX_PATH = "dish_ann_index.faiss"
VECTOR_INDEX_PATH = "dish_vector_index.json"

# Load the FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)

# Load dish names
with open(VECTOR_INDEX_PATH, "r") as f:
    dish_names = list(json.load(f).keys())

# FastAPI endpoint for embedding generation
API_URL = "http://127.0.0.1:8000/predict"  # Update this if hosted elsewhere

# Function to query the FAISS index
def query_faiss(embedding: np.ndarray, top_k: int = 5):
    """
    Query the FAISS index with an embedding and return the top_k closest matches.

    Parameters:
    ----------
    embedding : np.ndarray
        The embedding vector for the query.
    top_k : int, optional
        Number of closest matches to return (default is 5).

    Returns:
    -------
    list of tuples
        A list of (dish_name, distance) pairs sorted by similarity.
    """
    distances, indices = index.search(np.array([embedding]).astype("float32"), top_k)
    results = [(dish_names[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results


# Streamlit app
st.title("Dish Recommendation System (API-Based)")
st.write("Enter a dish name to find similar dishes!")

# Input from the user
user_input = st.text_input("Enter a dish name:", "")

# Number of results to display
top_k = st.slider(
    "Number of recommendations to show:", min_value=1, max_value=10, value=5
)

if user_input:
    st.write(f"Searching for dishes similar to: **{user_input}**")

    # Call the API to get the embedding
    try:
        response = requests.post(API_URL, json={"sentences": [user_input]})
        response.raise_for_status()
        embeddings = response.json()["embeddings"]

        # Check if embeddings were returned
        if not embeddings or len(embeddings[0]) == 0:
            st.error("Failed to retrieve embeddings for the input.")
        else:
            # Query the FAISS index
            embedding = np.array(embeddings[0])
            results = query_faiss(embedding, top_k=top_k)

            # Display the results
            st.write("### Recommendations:")
            for rank, (dish_name, distance) in enumerate(results, 1):
                st.write(f"{rank}. **{dish_name}** (Distance: {distance:.4f})")

    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the embedding API: {e}")
