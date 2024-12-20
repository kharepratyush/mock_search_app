
# Semantic Search System with Sentence Transformers, FAISS, and Streamlit

This repository provides an end-to-end solution for building a semantic search system using Sentence Transformers, FAISS, and Streamlit. The workflow includes data preparation, model training (both unsupervised and supervised), API deployment, and a user-friendly search interface.

---

## Table of Contents

- [Features](#features)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Components Overview](#components-overview)
- [Contributing](#contributing)
- [License](#license)

---

## Features

1. **Two-Stage Training**:
   - Unlabeled data training with **Multiple Negatives Ranking Loss (MNLR)**.
   - Fine-tuning on labeled data using **Cosine Similarity Loss**.

2. **Efficient Search**:
   - **FAISS** (Facebook AI Similarity Search) for Approximate Nearest Neighbor (ANN) search.

3. **User-Friendly Interface**:
   - A **Streamlit** app for querying similar items.

4. **API Deployment**:
   - Model deployed as a REST API using **FastAPI**.

---

## Folder Structure

```plaintext
.
├── data_loader.py                 # Load datasets with Hugging Face Datasets
├── unlabeled_dataset_creation.py  # Generate unlabeled query-dish pairs
├── labeled_dataset_creation.py    # Create labeled query-dish pairs
├── vector_index.py                # Generate embeddings and build FAISS index
├── ann.py                         # FAISS-based ANN querying
├── deploy_model.py                # FastAPI app for model deployment
├── streamlit_app.py               # Streamlit app for querying similar items
├── two_tower_mrnl.py              # Training script with MNLR and Cosine Similarity Loss
├── requirements.txt               # Dependencies for the project
├── README.md                      # Project documentation
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### **1. Generate Datasets**

Run the dataset creation scripts:
- Unlabeled dataset:
  ```bash
  python unlabeled_dataset_creation.py
  ```
- Labeled dataset:
  ```bash
  python labeled_dataset_creation.py
  ```

### **2. Train the Model**

Train the Sentence Transformer model:
```bash
python two_tower_mrnl.py
```

### **3. Build the Vector Index**

Generate embeddings and build the FAISS index:
```bash
python vector_index.py
```

### **4. Deploy the API**

Deploy the trained model as a REST API:
```bash
python deploy_model.py
```

### **5. Run the Streamlit App**

Launch the search interface:
```bash
streamlit run streamlit_app.py
```

---

## Components Overview

### **Data Preparation**
- `unlabeled_dataset_creation.py`: Generates a dataset of query-dish pairs without explicit labels.
- `labeled_dataset_creation.py`: Creates labeled query-dish pairs for supervised fine-tuning.

### **Model Training**
- `two_tower_mrnl.py`: Implements two-stage training:
  - **Stage 1**: Train on unlabeled data with MNLR.
  - **Stage 2**: Fine-tune on labeled data with Cosine Similarity Loss.

### **FAISS Indexing**
- `vector_index.py`: Generates vector embeddings for dishes and builds a FAISS index for ANN search.

### **API Deployment**
- `deploy_model.py`: Deploys the trained model as a REST API using FastAPI.

### **Streamlit Search App**
- `streamlit_app.py`: Interactive app for querying dishes and viewing recommendations.

---

## Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit changes and open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
