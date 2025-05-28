import uuid
import subprocess
import os
import torch
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from qdrant_client import QdrantClient, models
from langchain_openai import ChatOpenAI
import gradio as gr
import logging
import time
from typing import List, Tuple, Generator
from dataclasses import dataclass
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from queue import Queue
from threading import Thread
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Message:
    role: str
    content: str
    timestamp: str

class ChatHistory:
    def __init__(self):
        self.messages: List[Message] = []

    def add_message(self, role: str, content: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.messages.append(Message(role=role, content=content, timestamp=timestamp))

    def get_formatted_history(self, max_messages: int = 10) -> str:
        recent_messages = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        formatted_history = "\n".join([
            f"{msg.role}: {msg.content}" for msg in recent_messages
        ])
        return formatted_history

    def clear(self):
        self.messages = []

# Load environment variables and setup
load_dotenv()


OPENAPI_KEY = os.getenv("OPENAPI_KEY")
CHUTES_KEY = os.getenv("CHUTES_KEY")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=True,
        timeout=3000  # Timeout in seconds
    )
except Exception as e:
    logger.error("Failed to connect to Qdrant.")
    exit(1)

# Create the main collection for Mawared HR
collection_name = os.getenv("COLLECTION_NAME")

max_retries = 3
retry_count = 0
while retry_count < max_retries:
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            ),
            timeout=3000  # Timeout in seconds for collection creation
        )
        logger.info(f"Collection '{collection_name}' created successfully")
        break
    except Exception as e:
        retry_count += 1
        if retry_count == max_retries:
            if "already exists" not in str(e):
                logger.error(f"Error creating collection after {max_retries} attempts: {e}")
                exit(1)
            else:
                logger.info(f"Collection '{collection_name}' already exists")
                break
        logger.warning(f"Attempt {retry_count} failed, retrying in 2 seconds... Error: {e}")
        time.sleep(2)  # Wait between retries

db = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embeddings,
)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Create a new collection for logs
logs_collection_name = os.getenv("LOGS_COLLECTION_NAME")

# Check if logs collection exists
try:
    collections = client.get_collections().collections
    logs_collection_exists = any(collection.name == logs_collection_name for collection in collections)
except Exception as e:
    logger.error(f"Error checking collections: {e}")
    exit(1)

if not logs_collection_exists:
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            client.create_collection(
                collection_name=logs_collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # Same size as embeddings
                    distance=models.Distance.COSINE
                ),
                timeout=3000  # Timeout in seconds for collection creation
            )
            logger.info(f"Created new Qdrant collection: {logs_collection_name}")
            break
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                logger.error(f"Error creating logs collection after {max_retries} attempts: {e}")
                exit(1)
            logger.warning(f"Attempt {retry_count} failed, retrying in 2 seconds... Error: {e}")
            time.sleep(2)  # Wait 2 seconds before retrying
else:
    logger.info(f"Using existing logs collection: {logs_collection_name}")

def log_to_qdrant(question: str, answer: str):
    """Logs the question and answer to the Qdrant logs collection."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "question": question,
            "answer": answer,
            "timestamp": timestamp
        }

        # Convert the log entry to a vector (using embeddings)
        log_vector = embeddings.embed_documents([str(log_entry)])[0]

        # Generate a valid 64-bit unsigned integer ID
        valid_id = uuid.uuid4().int & (1 << 64) - 1  # Ensure it's a 64-bit unsigned integer

        # Insert the log into the Qdrant collection
        client.upsert(
            collection_name=logs_collection_name,
            points=[
                models.PointStruct(
                    id=valid_id,  # Use a valid 64-bit unsigned integer ID
                    vector=log_vector,
                    payload=log_entry
                )
            ]
        )
        logger.info(f"Logged question and answer to Qdrant collection: {logs_collection_name}")
    except Exception as e:
        logger.error(f"Failed to log to Qdrant: {e}")

#llm = ChatGoogleGenerativeAI(
    #model="gemini-2.0-flash-thinking-exp-01-21",
    #temperature=0.3,
    #max_tokens=None,
    #timeout=None,
    #max_retries=2,
    #api_key=GEMINI,
    #stream=True,
#)



llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3-0324",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=3,
    # api_key=OPENAPI_KEY,# if you prefer to pass api key in directly instaed of using env vars
    # base_url="https://openrouter.ai/api/v1",
    stream=True,
    api_key=CHUTES_KEY,
    base_url="https://llm.chutes.ai/v1/",

)

template = """

# Maxwell AI Assistant Guidelines

ADD SYSTEM PROMPT HERE
By adhering to these principles and guidelines, ensure every response is accurate, professional, and easy to follow.

Previous Conversation: {chat_history}
Retrieved Context: {context}
Current Question: {question}
Answer : {{answer}}

"""

prompt = ChatPromptTemplate.from_template(template)

def create_rag_chain(chat_history: str):
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "chat_history": lambda x: chat_history
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

chat_history = ChatHistory()

def process_stream(stream_queue: Queue, history: List[List[str]]) -> Generator[List[List[str]], None, None]:
    """Process the streaming response and update the chat interface"""
    current_response = ""

    while True:
        chunk = stream_queue.get()
        if chunk is None:  # Signal that streaming is complete
            break

        current_response += chunk
        new_history = history.copy()
        new_history[-1][1] = current_response  # Update the assistant's message
        yield new_history


def ask_question_gradio(question: str, history: List[List[str]]) -> Generator[tuple, None, None]:
    try:
        if history is None:
            history = []

        chat_history.add_message("user", question)
        formatted_history = chat_history.get_formatted_history()
        rag_chain = create_rag_chain(formatted_history)

        # Update history with user message and empty assistant message
        history.append([question, ""])  # User message

        # Create a queue for streaming responses
        stream_queue = Queue()

        # Function to process the stream in a separate thread
        def stream_processor():
            try:
                for chunk in rag_chain.stream(question):
                    stream_queue.put(chunk)
                stream_queue.put(None)  # Signal completion
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                stream_queue.put(None)

        # Start streaming in a separate thread
        Thread(target=stream_processor).start()

        # Yield updates to the chat interface
        response = ""
        for updated_history in process_stream(stream_queue, history):
            response = updated_history[-1][1]
            yield "", updated_history

        # Add final response to chat history
        chat_history.add_message("assistant", response)

        # Log the question and answer to Qdrant
        logger.info("Attempting to log question and answer to Qdrant")
        log_to_qdrant(question, response)

    except Exception as e:
        logger.error(f"Error during question processing: {e}")
        if not history:
            history = []
        history.append([question, "An error occurred. Please try again later."])
        yield "", history

def clear_chat():
    chat_history.clear()
    return [], ""

# Gradio Interface
with gr.Blocks() as iface:
    gr.Image("Image.jpg", width=800, height=200, show_label=False, show_download_button=False)
    gr.Markdown("# MaxWell  7.0.0 Deepseek")
    gr.Markdown('### Patch notes')
    gr.Markdown("""
**Ultra Fast Deepseek**

    """)

    chatbot = gr.Chatbot(
        height=750,
        show_label=False,
        bubble_full_width=False,
    )

    with gr.Row():
        with gr.Column(scale=20):
            question_input = gr.Textbox(
                label="Ask a question:",
                placeholder="Type your question here...",
                show_label=False
            )
        with gr.Column(scale=4):
            with gr.Row():
                with gr.Column():
                    send_button = gr.Button("Send", variant="primary", size="sm")
                    clear_button = gr.Button("Clear Chat", size="sm")

    # Handle both submit events (Enter key and Send button)
    submit_events = [question_input.submit, send_button.click]
    for submit_event in submit_events:
        submit_event(
            ask_question_gradio,
            inputs=[question_input, chatbot],
            outputs=[question_input, chatbot]
        )

    clear_button.click(
        clear_chat,
        outputs=[chatbot, question_input]
    )

if __name__ == "__main__":
    iface.launch()
