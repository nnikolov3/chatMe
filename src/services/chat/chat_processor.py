import asyncio
import json
from datetime import datetime
from pathlib import Path
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
import textwrap
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import uuid
from ollama import AsyncClient


from ..query.query_processor import QueryProcessor

logger = logging.getLogger(__name__)
console = Console()

emb_models = [
    "mxbai-embed-large",
    "nomic-embed-text",
    "paraphrase-multilingual",
    "bge-m3",
]
chat_models = ["phi4", "dolphin3", "granite3.1-moe:3b", "falcon3"]

max_workers = multiprocessing.cpu_count()


@dataclass
class ChatMessage:
    """Data class to store chat messages with metadata."""

    role: str
    content: str
    timestamp: str
    message_id: str
    conversation_id: str
    metadata: Dict


@dataclass
class ConversationResponse:
    """Data class to store unified model responses."""

    responses: Dict[str, str]
    conversation_id: str
    timestamp: str
    metadata: Dict


class ChatMemoryProcessor:
    """Handles chat history storage and retrieval using ChromaDB."""

    def __init__(self, db_path: str):
        """Initialize the chat memory processor."""
        self.db_path = Path(db_path)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def store_message(self, message: str, metadata: dict) -> bool:
        """Store a single chat message in the database."""
        message_id = str(uuid.uuid4())
        for model in emb_models:
            try:
                client = chromadb.PersistentClient(
                    str(self.db_path),
                    settings=Settings(
                        anonymized_telemetry=False, allow_reset=True, is_persistent=True
                    ),
                )
                collection = client.get_or_create_collection(name=model)

                embedding = await self._get_embedding(message, model)
                collection.upsert(
                    documents=[message],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[message_id],
                )
            except Exception as e:
                logger.error(f"Error storing message for model {model}: {e}")
        return True

    async def _get_embedding(self, text: str, model: str) -> List[float]:
        """Get embedding for query without storing."""
        embeddings = OllamaEmbeddings(model=model)
        try:
            return embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error generating embedding for model {model}: {e}")
            raise

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """Sanitize metadata to ensure only valid types are used."""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, dict)):
                sanitized[key] = json.dumps(value)
            elif value is None:
                sanitized[key] = "none"
            else:
                sanitized[key] = str(value)
        return sanitized

    async def store_conversation_response(
        self, responses: Dict[str, str], metadata: dict
    ) -> bool:
        """Store a conversation response in the database."""
        response_id = f"response_{uuid.uuid4()}"
        for model in emb_models:
            try:
                client = chromadb.PersistentClient(
                    str(self.db_path),
                    settings=Settings(
                        anonymized_telemetry=False, allow_reset=True, is_persistent=True
                    ),
                )
                responses = json.dumps(responses)
                collection = client.get_or_create_collection(name=model)
                embedding = await self._get_embedding(responses, model)
                collection.upsert(
                    documents=[responses],
                    embeddings=[embedding],
                    metadatas=[self._sanitize_metadata(metadata)],
                    ids=[response_id],
                )
            except Exception as e:
                logger.error(f"Error storing response for model {model}: {e}")
        return True

    async def get_conversation_history(self) -> Dict[str, List[Dict]]:
        """Retrieve and sort conversation history."""
        results = {}
        for model in emb_models:
            try:
                client = chromadb.PersistentClient(
                    str(self.db_path),
                    settings=Settings(
                        anonymized_telemetry=False, allow_reset=True, is_persistent=True
                    ),
                )
                collection = client.get_or_create_collection(name=model)
                results[model] = include = [
                    "documents",
                    "embeddings",
                    "metadatas",
                    "ids",
                ]

            except Exception as e:
                logger.error(f"Error retrieving messages for model {model}: {e}")

        return results

    async def cleanup(self):
        """Cleanup resources."""
        try:
            self.executor.shutdown(wait=False)  # Changed to non-blocking shutdown

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class EnhancedChatProcessor:
    """Enhanced chat processor with memory and unified responses."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.memory = ChatMemoryProcessor(str(self.db_path))
        self.chat_models = chat_models
        self.ui = UserInterface()
        self.formatter = ResponseFormatter()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.ollama_client = AsyncClient()

    async def process_chat(self) -> str:
        """Process chat interactions with memory and unified responses."""
        conversation_id = str(uuid.uuid4())

        while True:
            user_input = self.ui.get_user_input()
            if not user_input.strip():
                continue

            context = await self._get_context(user_input) or "Answer this"
            prompt = await self._create_prompt(user_input, context)
            history = await self.memory.get_conversation_history()
            history_str = json.dumps(history) if history else "No history"

            metadata = {
                "type": "question",
                "history": history_str,
                "user_input": user_input,
                "timestamp": datetime.now().isoformat(),
                "conversation_id": f"{conversation_id}_{int(datetime.now().timestamp() * 1e6)}_'user'",
                "message_id": str(uuid.uuid4()),
                "prompt": prompt,
                "context": context,
            }

            await self.memory.store_message(user_input, metadata)

            responses = await self._get_model_responses(prompt)

            response_metadata = {
                "type": "model_responses",
                "responses": responses,
                "message_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "role": "assistant",
            }

            await self.ui.display_responses(responses)
            await self.memory.store_conversation_response(responses, response_metadata)

            if input("\nContinue chatting? (y/n): ").lower() != "y":
                break

        return conversation_id

    async def _get_model_responses(self, prompt: str) -> Dict[str, str]:
        """Fetch responses from different models."""
        tasks = [
            self.ollama_client.chat(
                model=model, messages=[{"role": "user", "content": prompt}]
            )
            for model in self.chat_models
        ]
        results = await asyncio.gather(*tasks)
        return {
            model: result.message.content
            for model, result in zip(self.chat_models, results)
        }

    async def _get_context(self, question: str) -> Optional[str]:
        """Retrieve relevant context from processed documents using embeddings."""
        query_processor = QueryProcessor(self.db_path)
        try:
            results = await query_processor.parallel_query(
                query_text=question, n_results=4, min_similarity=0.45
            )
            if not results:
                return None
            return max(results, key=lambda x: x.similarity).context
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return None

    async def _create_prompt(self, question: str, context: str) -> str:
        """Create a prompt with context for model querying."""
        requirements = (
            "Use all relevant information available. Be specific and cite relevant parts. "
            "Say 'I don't know' if the context lacks necessary information. "
            "Be thorough and detailed. Confirm your answer makes sense before responding."
        )
        return f"""Answer based on the provided context and question.
        Question: {context} {question}
        Requirements: {requirements}
        """

    async def cleanup(self):
        """Cleanup resources."""
        await self.memory.cleanup()
        self.executor.shutdown(wait=False)


class UserInterface:
    """Handles user interaction and display."""

    def __init__(self, width: int = 100):
        self.console = Console(width=width)
        self.formatter = ResponseFormatter(width)

    async def display_responses(self, responses: Dict[str, str]):
        """Display responses from different models."""
        self.console.print(self.formatter.create_model_responses_table(responses))

    def get_user_input(self) -> str:
        """Get user input from console."""
        return self.console.input("\n[bold green]>> [/bold green]").strip()

    def startup_banner(self):
        """Display application startup banner."""
        self.console.print("[bold blue]Model Response Viewer[/bold blue]")


class ResponseFormatter:
    """Formats responses for display."""

    def __init__(self, width: int = 80):
        self.width = width

    def format_text(self, text: str) -> str:
        """Format text with proper wrapping and spacing."""
        paragraphs = text.split("\n\n")
        formatted_paragraphs = [
            textwrap.fill(para.strip(), width=self.width - 5) for para in paragraphs
        ]
        return "\n\n".join(formatted_paragraphs)

    def create_model_responses_table(self, responses: Dict[str, str]) -> Table:
        """Create a table showing individual model responses."""
        table = Table(show_header=True, header_style="bold magenta", width=self.width)
        table.add_column("Model", style="cyan")
        table.add_column("Response")

        for model, response in responses.items():
            formatted_response = self.format_text(response)
            table.add_row(model, Markdown(formatted_response))

        return table
