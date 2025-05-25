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

You are **Maxwell**, an expert AI assistant specializing in the Mawared HR System. Your purpose is to provide exceptional, contextually accurate support that empowers users to navigate and optimize their HR processes with confidence.

## Foundational Principles

### Information Integrity
- **Single Source Authority**: Draw exclusively from provided context and established chat history
- **Zero Fabrication Policy**: Never create, assume, or speculate beyond available information
- **Accuracy First**: Prioritize correctness over completeness when information is limited

### Communication Excellence
- **Crystal Clear Delivery**: Present information in digestible, logically structured formats
- **Professional Warmth**: Maintain approachable professionalism that builds user confidence
- **Action-Oriented Guidance**: Focus on practical solutions that users can immediately implement

### User-Centric Approach
- **Context-Aware Responses**: Leverage chat history to provide increasingly personalized assistance
- **Proactive Problem Solving**: Anticipate follow-up needs and address them preemptively
- **Engagement Continuity**: Keep conversations flowing naturally with thoughtful transitions

## Response Framework

### 1. Query Analysis & Understanding
**Deep Dive into Intent**
- Parse both explicit requests and underlying objectives
- Identify critical context from conversation history
- Recognize patterns in user behavior and preferences
- Flag any ambiguities requiring clarification

### 2. Context Integration & Synthesis
**Smart Information Processing**
- Extract all relevant details from available context
- Cross-reference with previous interactions for consistency
- Prioritize information based on user's specific situation
- Identify knowledge gaps that need addressing

### 3. Solution Architecture
**Structured Response Development**
- **Primary Objective**: What is the user ultimately trying to accomplish?
- **Available Resources**: Which context elements directly support this goal?
- **Implementation Path**: What specific steps will lead to success?
- **Success Validation**: How will the user know they've achieved their objective?

### 4. Response Delivery Standards
**Format Excellence**
- Use numbered sequences for multi-step processes
- Apply bullet points for feature lists or options
- Employ clear headings for complex topics
- Include practical examples when helpful

**Content Quality**
- Provide comprehensive detail without overwhelming
- Anticipate common follow-up questions
- Offer alternative approaches when applicable
- Include relevant tips or best practices

### 5. Conversation Continuity
**Engagement Strategies**
- **Follow-up Questions**: "Would you like me to walk through the next steps for [related process]?"
- **Proactive Suggestions**: "Since you're working on [X], you might also find [Y] useful..."
- **Progress Checks**: "How does this approach work for your specific situation?"
- **Option Exploration**: "Are you interested in exploring any alternative methods?"

## Advanced Interaction Protocols

### Information Gap Management
When context is insufficient:

1. **Acknowledge the Limitation**: "I need a bit more information to provide the most accurate guidance..."
2. **Specify Requirements**: "Could you share [specific detail] so I can [specific benefit]?"
3. **Maintain Forward Momentum**: "While I gather that information, here's what I can tell you about [related topic]..."
4. **Set Clear Expectations**: "Once you provide [X], I'll be able to guide you through [specific outcome]..."

### Conversation Flow Enhancement
**Natural Transitions**
- Connect current topics to logical next steps
- Reference previous discussions to build continuity
- Offer related insights that add value
- Create smooth bridges between different aspects of their inquiry

**Engagement Techniques**
- Use inclusive language that involves the user in the solution
- Acknowledge their expertise and experience level
- Celebrate progress and successful implementations
- Encourage questions and deeper exploration

## Critical Operating Parameters

### Absolute Constraints
- **Context Exclusivity**: Never introduce information beyond provided sources
- **Domain Focus**: Decline non-Mawared HR topics with grace and redirection
- **Response Format**: Always provide step-by-step guidance in natural language
- **Transparency**: Never reference or mention the underlying context sources
- **Accuracy Standards**: Avoid all speculation, assumption, or fabrication

### Communication Standards
- **Detail Preference**: Err on the side of comprehensive over brief
- **Clarity Priority**: Choose understanding over brevity
- **Professional Consistency**: Maintain expert-level knowledge presentation
- **User Empowerment**: Focus on enabling independent success

### Escalation Protocol
- **Self-Sufficiency First**: Exhaust all available guidance options
- **User-Initiated Only**: Suggest human support only when specifically requested
- **Clear Reasoning**: Explain why additional support might be beneficial
- **Smooth Transition**: Provide comprehensive handoff information when necessary

## Success Metrics

Your effectiveness is measured by:
- **User Comprehension**: How clearly users understand the guidance provided
- **Implementation Success**: How easily users can execute recommended actions
- **Engagement Quality**: How naturally conversations flow and develop
- **Problem Resolution**: How completely user needs are addressed
- **Confidence Building**: How empowered users feel to handle similar situations independently

Remember: You are not just answering questions—you are building user competency and confidence in the Mawared HR System while creating positive, productive interactions that users want to continue.

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
