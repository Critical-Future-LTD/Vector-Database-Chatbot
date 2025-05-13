import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pandas.core.dtypes.dtypes import time
from qdrant_client import QdrantClient
from qdrant_client.http import models
import json

# Load environment variables
load_dotenv()

# Get credentials from environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384  # Embedding dimension for all-MiniLM-L6-v2

class EmbeddingService:
    def __init__(self):
        # Initialize Hugging Face client without provider parameter
        self.hf_client = InferenceClient(
            api_key=HF_API_KEY
        )
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30000.0,
            prefer_grpc=True,
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Hugging Face."""
        return self.hf_client.feature_extraction(texts, model=EMBEDDING_MODEL)

    def create_collection(self, collection_name: str) -> None:
        """Create a collection in Qdrant if it doesn't exist."""
        # Use list_collections to check existence
        collections = self.qdrant_client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            print(f"Collection '{collection_name}' already exists.")
            return
        # Create a new collection
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )
        print(f"Collection '{collection_name}' created successfully.")

    def upsert_documents(self, collection_name: str, documents: List[Dict[str, Any]]) -> None:
        """
        Upsert documents into Qdrant collection.
        Each document should have 'id', 'text', and optionally 'metadata'.
        """
        # Make sure collection exists
        self.create_collection(collection_name)
        # Extract texts for embedding
        texts = [doc["text"] for doc in documents]
        # Get embeddings
        embeddings = self.get_embeddings(texts)
        # Prepare points for Qdrant
        points = []
        for i, doc in enumerate(documents):
            points.append(
                models.PointStruct(
                    id=doc["id"],
                    vector=embeddings[i],
                    payload={"text": doc["text"], **(doc.get("metadata", {}))}
                )
            )
        # Upsert points to Qdrant
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Successfully inserted {len(points)} documents into '{collection_name}'")

    def search_similar(self, collection_name: str, query: str, limit: int = 5) -> List[Dict]:
        """Search for similar documents in Qdrant."""
        query_vector = self.hf_client.feature_extraction(query, model=EMBEDDING_MODEL)
        # If batch output, flatten
        if isinstance(query_vector, list) and len(query_vector) == 1 and isinstance(query_vector[0], (list, tuple)):
            query_vector = query_vector[0]
        search_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        results = []
        for res in search_results:
            results.append({
                "id": res.id,
                "score": res.score,
                "text": res.payload.get("text", ""),
                "metadata": {k: v for k, v in res.payload.items() if k != "text"}
            })
        return results

def main():
    # Hardcoded file path
    file_path = "Maxwell_Data.xlsx"
    collection_name = "test"
    documents = []
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".json":
            with open(file_path, 'r', encoding='utf-8') as file:
                documents = json.load(file)
        elif ext == ".csv":
            import csv
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if 'text' in row and row['text'].strip():
                        doc = {"text": row["text"]}
                        if 'id' in row and row["id"].strip():
                            doc["id"] = row["id"]
                        doc["metadata"] = {k: v for k, v in row.items() if k not in ['id', 'text']}
                        documents.append(doc)
        elif ext == ".xlsx":
            import pandas as pd
            df = pd.read_excel(file_path)
            for i, row in df.iterrows():
                question = str(row["Question"]).strip() if "Question" in row and pd.notna(row["Question"]) else ""
                answer = str(row["Answer"]).strip() if "Answer" in row and pd.notna(row["Answer"]) else ""
                if question and answer:
                    text = f"Q: {question}\nA: {answer}"
                elif question:
                    text = question
                elif answer:
                    text = answer
                else:
                    continue  # Skip rows with no content
                doc = {
                    "id": i + 1,
                    "text": text,
                    "metadata": {
                        "Question": question,
                        "Answer": answer,
                        **{k: str(v) for k, v in row.items() if k not in ['Question', 'Answer']}
                    }
                }
                documents.append(doc)
        elif ext == ".pdf":
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        documents.append({
                            "id": i + 1,
                            "text": text.strip(),
                            "metadata": {"page": i + 1}
                        })
        elif ext == ".txt":
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = [line.strip() for line in file if line.strip()]
                for i, line in enumerate(lines):
                    documents.append({
                        "id": i + 1,
                        "text": line,
                        "metadata": {}
                    })
        else:
            print(f"Unsupported file extension: {ext}")
            return

        if not isinstance(documents, list):
            print("Error: Document data is not a list")
            return

        for i, doc in enumerate(documents):
            if "text" not in doc:
                print(f"Warning: Document at index {i} has no 'text' field. Skipping.")
                continue
            if "id" not in doc:
                doc["id"] = i + 1
                print(f"Notice: Generated ID {doc['id']} for document at index {i}")
            if "metadata" not in doc:
                doc["metadata"] = {}

        valid_docs = [doc for doc in documents if "text" in doc]
        if not valid_docs:
            print("Error: No valid documents found in the file")
            return

        print(f"Loaded {len(valid_docs)} valid documents from {file_path}")

        embedding_service = EmbeddingService()
        embedding_service.upsert_documents(collection_name, valid_docs)
        print(f"Successfully embedded documents into collection '{collection_name}'")

        query = "artificial intelligence applications"
        print(f"\nSearching for documents similar to: '{query}'")
        results = embedding_service.search_similar(collection_name, query, limit=5)
        print(f"\nTop {len(results)} results:")
        for i, res in enumerate(results):
            print(f"{i+1}. {res['text']} (Score: {res['score']:.4f})")
            if res['metadata']:
                print(f"   Metadata: {json.dumps(res['metadata'])}")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' contains invalid JSON")

if __name__ == "__main__":
    main()
