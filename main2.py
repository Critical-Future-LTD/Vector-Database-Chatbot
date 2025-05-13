import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
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
        # Initialize Hugging Face client
        self.hf_client = InferenceClient(
            provider="hf-inference",
            api_key=HF_API_KEY,
        )
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Hugging Face."""
        embeddings = []
        
        # Process texts in batches to avoid potential API limitations
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # For each text in the batch, get its embedding
            batch_embeddings = []
            for text in batch:
                result = self.hf_client.sentence_similarity(
                    inputs={
                        "source_sentence": text,
                        "sentences": [text]  # We only need the embedding vector
                    },
                    model=EMBEDDING_MODEL,
                )
                # The similarity score with itself should be 1.0, so we can extract the vector
                vector = result[0]["embedding"]
                batch_embeddings.append(vector)
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def create_collection(self, collection_name: str) -> None:
        """Create a collection in Qdrant if it doesn't exist."""
        try:
            self.qdrant_client.get_collection(collection_name)
            print(f"Collection '{collection_name}' already exists.")
        except Exception:
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
        # Get query embedding
        query_result = self.hf_client.sentence_similarity(
            inputs={
                "source_sentence": query,
                "sentences": [query]
            },
            model=EMBEDDING_MODEL,
        )
        query_vector = query_result[0]["embedding"]
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        # Format results
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
    file_path = "data/documents.json"
    collection_name = "semantic_search"
    
    # Initialize the embedding service
    embedding_service = EmbeddingService()
    
    try:
        # Load documents from the specified file
        with open(file_path, 'r', encoding='utf-8') as file:
            documents = json.load(file)
        
        # Validate the document format
        if not isinstance(documents, list):
            print("Error: File must contain a JSON array of documents")
            return
            
        for i, doc in enumerate(documents):
            if "text" not in doc:
                print(f"Warning: Document at index {i} has no 'text' field. Skipping.")
                continue
            if "id" not in doc:
                # Generate an ID if not provided
                doc["id"] = i + 1
                print(f"Notice: Generated ID {doc['id']} for document at index {i}")
            if "metadata" not in doc:
                doc["metadata"] = {}
        
        # Filter out invalid documents
        valid_docs = [doc for doc in documents if "text" in doc]
        
        if not valid_docs:
            print("Error: No valid documents found in the file")
            return
            
        print(f"Loaded {len(valid_docs)} valid documents from {file_path}")
        
        # Create collection and insert documents
        embedding_service.upsert_documents(collection_name, valid_docs)
        print(f"Successfully embedded documents into collection '{collection_name}'")
        
        # Hardcoded search query (comment out if you don't want to search)
        query = "artificial intelligence applications"
        print(f"\nSearching for documents similar to: '{query}'")
        results = embedding_service.search_similar(collection_name, query, limit=5)
        
        # Print results
        print(f"\nTop {len(results)} results:")
        for i, res in enumerate(results):
            print(f"{i+1}. {res['text']} (Score: {res['score']:.4f})")
            if res['metadata']:
                print(f"   Metadata: {json.dumps(res['metadata'])}")
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        print("Make sure to create the 'data' directory and place your JSON file there")
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' contains invalid JSON")


if __name__ == "__main__":
    main()
