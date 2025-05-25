import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import json
import pandas as pd
import pdfplumber
import csv

# Load environment variables
load_dotenv()

# Get credentials from environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384  # Embedding dimension for all-MiniLM-L6-v2

class EmbeddingService:
    def __init__(self):
        # Initialize HuggingFace embeddings (similar to second code)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60000,  # Increased timeout to 60 seconds
            prefer_grpc=True,
        )

        # Initialize vectorstore as None, will be created when needed
        self.vectorstore = None

    def create_collection(self, collection_name: str) -> None:
        """Create a collection in Qdrant if it doesn't exist."""
        max_retries = 3
        retry_delay = 2  # seconds

        # Check collection existence with retries
        for attempt in range(max_retries):
            try:
                collections = self.qdrant_client.get_collections().collections
                collection_names = [collection.name for collection in collections]

                if collection_name in collection_names:
                    print(f"Collection '{collection_name}' already exists.")
                    return
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to check collection existence after {max_retries} attempts: {str(e)}")
                print(f"Attempt {attempt + 1} failed to check collections, retrying in {retry_delay}s...")
                time.sleep(retry_delay)

        # Create a new collection with retries
        for attempt in range(max_retries):
            try:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=VECTOR_SIZE,
                        distance=models.Distance.COSINE
                    ),
                    timeout=6000  # 60 seconds timeout for creation
                )
                print(f"Collection '{collection_name}' created successfully.")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to create collection after {max_retries} attempts: {str(e)}")
                print(f"Attempt {attempt + 1} failed to create collection, retrying in {retry_delay}s...")
                time.sleep(retry_delay)

    def create_vectorstore(self, collection_name: str) -> Qdrant:
        """Create and return the LangChain Qdrant vectorstore."""
        # Ensure collection exists
        self.create_collection(collection_name)

        # Create and return the vector store (similar to second code structure)
        vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name=collection_name,
            embeddings=self.embeddings
        )

        return vectorstore

    def load_documents_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load documents from various file formats and convert to LangChain-like format."""
        documents = []
        ext = os.path.splitext(file_path)[1].lower()

        print(f"Loading documents from {file_path}...")

        try:
            if ext == ".json":
                with open(file_path, 'r', encoding='utf-8') as file:
                    raw_docs = json.load(file)
                    for i, doc in enumerate(raw_docs):
                        if isinstance(doc, dict) and "text" in doc:
                            documents.append({
                                "page_content": doc["text"],
                                "metadata": {
                                    "id": doc.get("id", i + 1),
                                    "source": file_path,
                                    **doc.get("metadata", {})
                                }
                            })

            elif ext == ".csv":
                with open(file_path, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    for i, row in enumerate(reader):
                        if 'text' in row and row['text'].strip():
                            documents.append({
                                "page_content": row["text"],
                                "metadata": {
                                    "id": row.get("id", i + 1),
                                    "source": file_path,
                                    **{k: v for k, v in row.items() if k not in ['id', 'text']}
                                }
                            })

            elif ext == ".xlsx":
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

                    documents.append({
                        "page_content": text,
                        "metadata": {
                            "id": i + 1,
                            "source": file_path,
                            "Question": question,
                            "Answer": answer,
                            **{k: str(v) for k, v in row.items() if k not in ['Question', 'Answer']}
                        }
                    })

            elif ext == ".pdf":
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text and text.strip():
                            documents.append({
                                "page_content": text.strip(),
                                "metadata": {
                                    "id": i + 1,
                                    "source": file_path,
                                    "page": i + 1
                                }
                            })

            elif ext == ".txt":
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = [line.strip() for line in file if line.strip()]
                    for i, line in enumerate(lines):
                        documents.append({
                            "page_content": line,
                            "metadata": {
                                "id": i + 1,
                                "source": file_path
                            }
                        })
            else:
                print(f"Unsupported file extension: {ext}")
                return []

        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found")
            return []
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            return []

        print(f"Loaded {len(documents)} documents from {file_path}")
        return documents

    def upsert_documents(self, collection_name: str, file_path: str) -> None:
        """
        Load documents from file and upsert them into Qdrant collection using LangChain structure.
        """
        # Load documents from file
        documents = self.load_documents_from_file(file_path)

        if not documents:
            print("No valid documents found to upsert")
            return

        # Create vectorstore
        vectorstore = self.create_vectorstore(collection_name)

        # Convert to LangChain Document format
        from langchain_core.documents import Document
        langchain_docs = []

        print("Converting documents to LangChain format...")
        for doc in tqdm(documents, desc="Converting docs"):
            langchain_docs.append(Document(
                page_content=doc["page_content"],
                metadata=doc["metadata"]
            ))

        # Add documents to the vector store (similar to second code)
        print("Adding documents to vector store...")
        vectorstore.add_documents(langchain_docs)

        print(f"Successfully inserted {len(langchain_docs)} documents into '{collection_name}'")

    def search_similar(self, collection_name: str, query: str, limit: int = 5) -> List[Dict]:
        """Search for similar documents using LangChain vectorstore."""
        # Create vectorstore if not exists
        if not self.vectorstore or self.vectorstore.collection_name != collection_name:
            self.vectorstore = self.create_vectorstore(collection_name)

        # Perform similarity search
        results = self.vectorstore.similarity_search_with_score(query, k=limit)

        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "id": doc.metadata.get("id", "unknown"),
                "score": score,
                "text": doc.page_content,
                "metadata": {k: v for k, v in doc.metadata.items() if k != "id"}
            })

        return formatted_results

def main():
    # Hardcoded file path
    file_path = "Maxwell_Data.xlsx"
    collection_name = COLLECTION_NAME or "default_collection"

    try:
        embedding_service = EmbeddingService()

        # Upsert documents using the new structure
        embedding_service.upsert_documents(collection_name, file_path)
        print(f"Successfully embedded documents into collection '{collection_name}'")

        # Test search functionality
        query = "artificial intelligence applications"
        print(f"\nSearching for documents similar to: '{query}'")
        results = embedding_service.search_similar(collection_name, query, limit=5)

        print(f"\nTop {len(results)} results:")
        for i, res in enumerate(results):
            print(f"{i+1}. {res['text'][:200]}... (Score: {res['score']:.4f})")
            if res['metadata']:
                print(f"   Metadata: {json.dumps(res['metadata'], indent=2)}")
            print()

    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
