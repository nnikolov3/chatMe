"""
Main chat module
"""

import asyncio
import json
import logging
import multiprocessing

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings
from ollama import AsyncClient
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from src.services.query.query_processor import QueryProcessor

logger = logging.getLogger(__name__)
console = Console()

chat_models = ["phi-15B-4-Q8", "nvidia_q8"]


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

    def __init__(self, db_path: str, emb_models: list):
        """Initialize the chat memory processor."""
        self.db_path = Path(db_path)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.emb_models = emb_models

    async def store_message(self, message: str, metadata: dict) -> bool:
        """Store a single chat message in the database."""
        message_id = str(uuid.uuid4())

        tasks = [
            self.process_storing(model, message, message_id, metadata)
            for model in self.emb_models
        ]
        await asyncio.gather(*tasks)

    async def process_storing(self, model, message, message_id, metadata):
        """Process Asynchronously in ChromaDB"""
        try:
            client = chromadb.PersistentClient(
                str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True,
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
            logger.error(
                "Error storing message for model %s -> %s", model, e
            )
        return True

    async def _get_embedding(self, text: str, model: str) -> List[float]:
        """Get embedding for query without storing."""
        embeddings = OllamaEmbeddings(model=model)
        try:
            return embeddings.embed_query(text)
        except Exception as e:
            logger.error(
                "Error generating embedding for model %s -> %s", model, e
            )
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


class EnhancedChatProcessor:
    """Enhanced chat processor with memory and unified responses."""

    def __init__(self, db_path: str, emb_models):
        self.db_path = Path(db_path)
        self.memory = ChatMemoryProcessor(str(self.db_path), emb_models)
        self.chat_models = chat_models
        self.ui = UserInterface()
        self.formatter = ResponseFormatter()
        self.ollama_client = AsyncClient()
        self.emb_models = emb_models
        self.new_chat = True

    async def process_chat(self) -> str:
        """Main chat loop with error handling and graceful shutdown"""
        conversation_id = str(uuid.uuid4())
        chat_data = []

        try:
            while True:
                user_input = self.ui.get_user_input()
                if not user_input.strip():
                    continue

                if self.new_chat is True:
                    context = ""
                else:
                    context = await self._get_context(user_input)
                    if not context:
                        context = ""

                prompt = await self._create_prompt(user_input, context)

                responses = await self._get_model_responses(prompt)
                message_id = str(uuid.uuid4())
                timestamp = str(datetime.now().isoformat())

                metadata = {
                    "type": "question",
                    "timestamp": timestamp,
                    "message_id": message_id,
                    "response_metadata": json.dumps(
                        {  # Convert to JSON string
                            "type": "model_responses",
                            "role": "assistant",
                        }
                    ),
                }

                await self.ui.display_responses(responses)
                data = {
                    "question": user_input,
                    "prompt": prompt,
                    "responses": responses,
                }

                chat_data.append({"data": data, "metadata": metadata})

                await self.memory.store_message(json.dumps(data), metadata)

                if self.new_chat is True:
                    self.new_chat = False

                if input("\nContinue chatting? (y/n): ").lower() != "y":
                    break

        except KeyboardInterrupt:
            print("\nChat interrupted by user. Saving chat data...")
        except Exception as e:
            print(
                f"\nAn error occurred: {str(e)}. Saving chat data to prevent loss..."
            )
        finally:
            # Save the chat data regardless of how the loop ended
            today_date = datetime.now().strftime("%Y-%m-%d")
            folder_path = os.path.join("src", "data", "chats")
            file_path = os.path.join(folder_path, f"{today_date}.json")

            # Ensure the directory exists
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Write the conversation to JSON file
            with open(file_path, "w") as json_file:
                json.dump(chat_data, json_file, indent=4)

        return conversation_id

    async def _get_model_responses(self, prompt: str) -> Dict[str, str]:
        """Fetch responses from different models."""
        tasks = [
            self.ollama_client.chat(
                model=model, messages=[{"role": "user", "content": prompt}]
            )
            for model in self.chat_models
        ]
        results = await asyncio.gather(
            *tasks
        )  # Use asyncio.gather for coroutines
        return {
            model: result.message.content
            for model, result in zip(self.chat_models, results)
        }

    async def _get_context(self, question: str) -> Optional[str]:
        """Retrieve relevant context from
        processed documents using embeddings."""
        query_processor = QueryProcessor(self.db_path, self.emb_models)
        try:
            results = await query_processor.parallel_query(
                query_text=question, n_results=1, min_similarity=0.55
            )
            if not results:
                return None

            return results

        except Exception as e:
            logger.error("Error retrieving context: %s", e)

            return None

    async def _create_prompt(self, question: str, context: str) -> str:
        """Create a prompt with context for model querying."""
        requirements = """Be detailed and thorough,use analogies,give practical examples,structure for understanding, ensure correctness and accuracy"""
        return f""" Question: {question}, Requirements: {requirements}, Context: {context}
        """


class UserInterface:
    """Handles user interaction and display."""

    def __init__(self, width: int = 100):
        self.console = Console(width=width)
        self.formatter = ResponseFormatter(width)

    async def display_responses(self, responses: Dict[str, str]):
        """Display responses from different models."""
        self.console.print(
            self.formatter.create_model_responses_table(responses)
        )

    def get_user_input(self) -> str:
        """Get user input from console."""
        return self.console.input("\n[bold green]>> [/bold green]").strip()

    def startup_banner(self):
        """Display application startup banner."""
        self.console.print("[bold blue]Model Response Viewer[/bold blue]")


class ResponseFormatter:
    """Formats responses for display."""

    def __init__(self, width: int = 100):
        self.width = width

    def format_text(self, text: str) -> str:
        """Format text with proper wrapping and spacing."""
        paragraphs = text.split("\n")
        return "\n".join(paragraphs)

    def create_model_responses_table(
        self, responses: Dict[str, str]
    ) -> Table:
        """Create a table showing individual model responses."""
        table = Table(
            show_header=True, header_style="bold magenta", width=self.width
        )
        table.add_column("Model", style="cyan")
        table.add_column("Response")

        for model, response in responses.items():
            table.add_row(model, Markdown(self.format_text(response)))

        return table
