"""
Main module for query processing
"""

import asyncio
import logging
import multiprocessing
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
)

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class QueryResult:
    """Data class to store query results with enhanced metadata."""

    model: str
    document_id: str
    content: str
    similarity: float
    metadata: Dict
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Convert the query result to a dictionary format."""
        res = {
            "model": self.model,
            "document_id": self.document_id,
            "content": self.content,
            "similarity": self.similarity,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
        return res


class QueryProcessor:
    """Enhanced query processor with improved error handling and caching."""

    def __init__(
        self,
        db_path: str,
        emb_models: list,
        max_workers: int = multiprocessing.cpu_count(),
    ):
        """Initialize the query processor with configurable worker pool.

        Args:
            db_path: Path to the ChromaDB database
            max_workers: Maximum number of worker threads for async operations
        """
        self.db_path = Path(db_path)
        self._validate_db_path()
        self.emb_models = emb_models

        # Initialize thread pool with configurable size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Initialize ChromaDB client with better error handling
        try:
            self.client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True,
                ),
            )

        except Exception as e:
            logger.error("Failed to initialize ChromaDB client: %s", e)
            raise

    def _validate_db_path(self) -> None:
        """Validate the database path exists and is accessible."""
        if not self.db_path.exists():
            self.db_path.mkdir(parents=True, exist_ok=True)

        elif not self.db_path.is_dir():
            raise ValueError(
                f"Database path {self.db_path} exists but is not a directory"
            )

    async def _get_collection(self, model: str) -> Optional[Collection]:
        """Safely get a collection for the specified model."""
        try:
            collection_names = self.client.list_collections()
            if model not in collection_names:
                logger.warning("Collection for model %s not found", model)
                return None

            collection = self.client.get_collection(name=model)
            if collection.count() == 0:
                logger.warning("Collection '%s' is empty", model)
                return None

            return collection

        except Exception as e:
            logger.error("Error accessing collection %s -> %s", model, e)
            return None

    async def _get_embedding(self, text: str, model: str) -> List[float]:
        """Get embedding for query without storing."""
        try:
            emb = OllamaEmbeddings(model=model)
            embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: emb.embed_documents([text])
            )
            return embedding[0]  # Return just the embedding vector
        except Exception as e:
            logger.error("Error generating query embedding: %s", e)
            raise

    async def query(
        self,
        query_text: str,
        model: str,
        n_results: int = 2,
        min_similarity: float = 0.60,
    ) -> List[QueryResult]:
        """Query only - no storage."""
        if not query_text.strip():
            return []

        try:
            # Get embedding without storage
            query_embedding = await self._get_embedding(
                query_text, model
            )  # Modified to not store
            collection = await self._get_collection(model)

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            matched_results = []

            if results["documents"] and results["documents"][0]:

                for doc, metadata, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):

                    similarity = 1 - distance
                    if similarity >= min_similarity:
                        matched_results.append(
                            QueryResult(
                                model=model,
                                document_id=metadata.get("id", "unknown"),
                                content=doc,
                                similarity=similarity,
                                metadata=metadata,
                            )
                        )

            return matched_results

        except Exception as e:
            logger.error("Error during query operation: %s", e)
            return []

    async def parallel_query(
        self,
        query_text: str,
        n_results: int = 2,
        min_similarity: float = 0.60,
    ) -> List[QueryResult]:
        """Execute queries across all
        available models in parallel and combine results.
        """
        models = self.emb_models
        if not models:
            logger.warning("No models available for querying")
            return []

        tasks = [
            self.query(
                query_text=query_text,
                model=model,
                n_results=n_results,
                min_similarity=min_similarity,
            )
            for model in models
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        combined_results = []
        for model_results in results:

            if isinstance(
                model_results, list
            ):  # Check if results are as expected
                combined_results.extend(model_results)
            else:
                logger.error(
                    "Unexpected result type from model query: %s",
                    model_results,
                )

        # Sort and return results based on similarity
        return sorted(
            combined_results, key=lambda x: x.similarity, reverse=True
        )

    def get_available_models(self) -> List[str]:
        """Get list of available models with collection statistics.

        Returns:
            List of model names with available collections
        """
        try:

            collection_names = self.client.list_collections()

            # Log detailed collection statistics
            for name in collection_names:
                try:
                    collection = self.client.get_collection(name=name)
                    if not collection:
                        logger.error("Error no collection found")

                except Exception as e:
                    logger.error(
                        "Error accessing collection %s -> %s", name, e
                    )

            return collection_names

        except Exception as e:
            logger.error("Error listing collections: %s", e)
            return []

    async def cleanup(self):
        """Cleanup resources with enhanced error handling."""
        try:

            # Shutdown thread pool
            self.executor.shutdown(wait=True)

        except Exception as e:
            logger.error("Error during cleanup: %s", e)
            raise


class QueryInterface:
    """Interface for interacting with the query processor."""

    def __init__(self, db_path: str, emb_models: list):
        """Initialize the query interface."""
        self.emb_models = emb_models
        self.processor = QueryProcessor(db_path, emb_models=self.emb_models)
        self._shutdown_event = asyncio.Event()

    async def setup(self):
        """Initialize the interface and set up signal handlers."""
        # Set up graceful shutdown handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self):
        """Handle shutdown signals gracefully."""
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
        self._shutdown_event.set()

    async def interactive_query(self):
        """Start an interactive query session."""
        await self.setup()

        console.print(
            "\n[cyan]Enter your queries (press Ctrl+C to exit):[/cyan]"
        )

        while not self._shutdown_event.is_set():
            try:
                # Get query from user
                query = await asyncio.get_event_loop().run_in_executor(
                    None, input, "\nQuery: "
                )

                if not query.strip():
                    continue

                # Execute parallel query with progress indication
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        "[cyan]Querying all models...", total=None
                    )

                    results = await self.processor.parallel_query(query)
                    progress.update(task, advance=1)

                # Display results
                if not results:
                    console.print(
                        "[yellow]No matching results found[/yellow]"
                    )
                    continue

                console.print("\n[green]Results:[/green]")
                for i, result in enumerate(results[:10], 1):
                    console.print(
                        f"\n[cyan]{i}. Model: {result.model}[/cyan]"
                        f"\nSimilarity: {result.similarity:.6%}"
                        f"\nDocument: {result.document_id}"
                        f"\nContent: {result.content[:11000]}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error processing query: %s", e)
                console.print(f"[red]Error: {str(e)}[/red]")

    async def cleanup(self):
        """Cleanup resources."""
        await self.processor.cleanup()


async def get_query_interface(
    db_path: str, emb_models: list
) -> QueryInterface:
    """Factory function to create and initialize a QueryInterface instance.

    Args:
        db_path: Path to the ChromaDB database

    Returns:
        Initialized QueryInterface instance
    """
    interface = QueryInterface(db_path, emb_models)
    await interface.setup()
    return interface
