"""
Main chat module
"""

import asyncio
import json
import logging
import multiprocessing

import os
import re
import sys
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

chat_models = [
    "DeepSeekR1_Q8v2",
]

USER = "NIKO"

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

        self.chat_path = Path("src/data/chats")

        try:
            while True:
                user_input = self.ui.get_user_input()
                if not user_input.strip():
                    continue
                mytime = datetime.now()

                time_s = mytime.strftime("%H:%M:%S")
                # Get today's date formatted as YYYY-MM-DD
                _date = datetime.now()
                today_date = _date.strftime("%Y-%m-%d")

                # Construct the file path for the JSON file
                json_file_path = os.path.join(
                    self.chat_path, f"{today_date}.json"
                )

                context_from_file = ""
                if not self.new_chat:
                    # Read the JSON file for today's date
                    if os.path.exists(json_file_path):
                        with open(json_file_path, "r") as file:
                            try:
                                context_from_file = json.load(file)

                            except json.JSONDecodeError:
                                print(
                                    f"Error decoding JSON from {json_file_path}"
                                )
                    else:
                        print(f"File not found: {json_file_path}")

                    # Get additional context, if any
                    context_from_function = await self._get_context(
                        user_input
                    )
                elif self.new_chat:

                    context_from_file = f"New chat starting at {time_s}"
                    context_from_function = (
                        f"Pay attention to previous conversations as we go"
                    )

                # Concatenate both contexts

                context = {
                    "today_chat": context_from_file,
                    "context_from_previous_chats": context_from_function,
                }
                prompt = await self._create_prompt(user_input, context)

                responses = await self._get_model_responses(prompt)

                message_id = str(uuid.uuid4())
                timestamp = str(datetime.now().isoformat())

                metadata = {
                    "type": "question",
                    "timestamp": timestamp,
                    "message_id": message_id,
                    "user": USER,
                    "response_metadata": json.dumps(
                        {  # Convert to JSON string
                            "type": "model_responses",
                            "role": "assistant",
                        }
                    ),
                }

                await self.ui.display_responses(responses)
                data = {
                    "question": {"user": USER, "user_input": user_input},
                    "responses": responses,
                    "time": time_s,
                }

                filtered_data = json.loads(
                    await self.filter_special_chars(json.dumps(data))
                )
                await self.memory.store_message(
                    str(filtered_data), metadata
                )

                chat_data = {"data": filtered_data, "metadata": metadata}

                await self._update_json_file(chat_data)

                if self.new_chat is True:
                    self.new_chat = False

                if input("\nContinue chatting? (y/n): ").lower() != "y":
                    break

        except KeyboardInterrupt:
            print("\nChat interrupted by user. Saving chat data...")
            await self._update_json_file(chat_data)
            sys.exit(1)
        except Exception as e:
            print(
                f"\nAn error occurred: {str(e)}. Saving chat data to prevent loss..."
            )
            await self._update_json_file(chat_data)
            sys.exit(1)

        return conversation_id

    async def _update_json_file(self, chat_data):
        today_date = datetime.now().strftime("%Y-%m-%d")
        folder_path = os.path.join("src", "data", "chats")
        file_path = os.path.join(folder_path, f"{today_date}.json")

        # Ensure the directory exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Load existing data or initialize an empty list
        existing_data = []
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as json_file:
                try:
                    # Load the file content, which should be a list
                    existing_data = json.load(json_file)
                    # Ensure it's a list
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
                except json.JSONDecodeError:
                    # If the file is empty or corrupted, start with an empty list
                    existing_data = []

        # Append new data
        existing_data.append(chat_data)

        # Write the updated data back to the file
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(existing_data, json_file, indent=4)

    async def filter_special_chars(self, text):
        """Clean up the gather text from too many spaces
        and useless special chars"""
        cleaned_text = re.sub(r"[\t\n\r]", " ", text)

        # Step 2: Replace multiple spaces with a single space
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        return cleaned_text

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
                query_text=question, n_results=2, min_similarity=0.6
            )
            if not results:
                return None

            return results

        except Exception as e:
            logger.error("Error retrieving context: %s", e)

            return None

    async def _create_prompt(self, question: str, context: str) -> str:
        """Create a prompt with context for model querying."""
        requirements = """You are my expert; Be detailed,thorough,correct, accurate. Make use of all available information """
        return f""" 'User': {USER} 'Context': {context},'Requirements': {requirements}, 'Question': {question}
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
            table.add_row(
                "",
                Markdown(
                    "=========================================================="
                ),
            )

        return table
