import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

# Define the FastAPI application
app = FastAPI(
    title="SentenceTransformer Model Deployment",
    description="A simple API to serve SentenceTransformer predictions.",
    version="1.0",
)

# Load the pre-trained model
MODEL_PATH = "models/final"  # Update this path to match your final model directory
try:
    model = SentenceTransformer(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    raise

# Define input and output schemas
class PredictionInput(BaseModel):
    sentences: List[str]


class PredictionOutput(BaseModel):
    embeddings: List[List[float]]


# API endpoint to get embeddings for input sentences
@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Generate embeddings for the given input sentences.

    Parameters:
    ----------
    input_data : PredictionInput
        JSON payload containing a list of sentences.

    Returns:
    -------
    PredictionOutput
        JSON response with the embeddings for each input sentence.
    """
    sentences = input_data.sentences

    if not sentences:
        raise HTTPException(status_code=400, detail="No sentences provided.")

    try:
        # Generate embeddings
        embeddings = model.encode(sentences, convert_to_tensor=False)

        embeddings_list = [
            embedding.tolist() if hasattr(embedding, "tolist") else embedding
            for embedding in embeddings
        ]

        return PredictionOutput(embeddings=embeddings_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {e}")


# Health check endpoint
@app.get("/health")
def health_check():
    """
    Simple health check endpoint.

    Returns:
    -------
    dict
        JSON response indicating the API is running.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
