"""
Resource management module for system monitoring and optimization.

This module provides classes for monitoring and managing system resources
including CPU, memory, and GPU utilization.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

# Third-party imports
import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings
from rich.console import Console

# Initialize logger and console
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class QueryResult:
    """Data class to store query results."""

    model: str
    document_id: str
    content: str
    distance: float
    metadata: Dict


class QueryProcessor:
    """Handles querying operations for the vector database."""

    def __init__(self, db_path: str, models: Optional[List[str]] = None):
        """Initialize the query processor.

        Args:
            db_path: Path to the ChromaDB database
            models: List of embedding models to query. If None, uses all available models.
        """
        self.db_path = Path(db_path)
        self.models = models or [
            "bge-m3",
            "paraphrase-multilingual",
            "mxbai-embed-large",
            "nomic-embed-text",
        ]
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False),
        )

    async def query(
        self,
        query_text: str,
        n_results: int = 5,
        min_similarity: float = 0.7,
        specific_model: Optional[str] = None,
    ) -> List[QueryResult]:
        """Query the vector database across all models or a specific model.

        Args:
            query_text: The text to search for
            n_results: Number of results to return per model
            min_similarity: Minimum similarity threshold (0-1)
            specific_model: Optional specific model to query

        Returns:
            List of QueryResult objects sorted by relevance
        """
        models_to_query = (
            [specific_model] if specific_model else self.models
        )
        all_results = []

        for model in models_to_query:
            try:
                # Get embeddings for the query
                loop = asyncio.get_event_loop()
                emb = OllamaEmbeddings(model=model)
                query_embedding = await loop.run_in_executor(
                    self.executor, lambda: emb.embed_documents([query_text])
                )

                # Get the collection for this model
                try:
                    collection = self.client.get_collection(name=model)
                except Exception as e:
                    logger.warning(
                        f"Collection not found for model {model}: {e}"
                    )
                    continue

                # Query the collection
                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"],
                )

                # Process results
                for idx, (doc, metadata, distance) in enumerate(
                    zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    )
                ):
                    # Convert distance to similarity score (1 - distance for cosine distance)
                    similarity = 1 - distance

                    if similarity >= min_similarity:
                        all_results.append(
                            QueryResult(
                                model=model,
                                document_id=metadata.get(
                                    "json_path", f"doc_{idx}"
                                ),
                                content=doc,
                                distance=distance,
                                metadata=metadata,
                            )
                        )

            except Exception as e:
                logger.error(f"Error querying model {model}: {e}")
                continue

        # Sort results by distance (lower is better)
        all_results.sort(key=lambda x: x.distance)
        return all_results

    def get_available_models(self) -> List[str]:
        """Get list of available models in the database."""
        collections = self.client.list_collections()
        return [collection.name for collection in collections]

    async def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a specific document by its ID."""
        for model in self.models:
            try:
                collection = self.client.get_collection(name=model)
                results = collection.get(
                    where={"json_path": doc_id},
                    include=["documents", "metadatas"],
                )
                if results["documents"]:
                    return {
                        "content": results["documents"][0],
                        "metadata": results["metadatas"][0],
                    }
            except Exception as e:
                logger.debug(f"Error getting document from {model}: {e}")
                continue
        return None

    async def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)


class QueryInterface:
    """User interface for querying the vector database."""

    def __init__(self, db_path: str):
        """Initialize the query interface.

        Args:
            db_path: Path to the ChromaDB database
        """
        self.processor = QueryProcessor(db_path)
        self.console = Console()

    async def interactive_query(self):
        """Run an interactive query session."""
        try:
            while True:
                # Get query parameters
                self.console.print(
                    "\n[cyan]Enter your query (or 'exit' to quit):[/cyan]"
                )
                query = input().strip()

                if query.lower() == "exit":
                    break

                self.console.print(
                    "[cyan]Number of results per model (default: 5):[/cyan]"
                )
                try:
                    n_results = int(input().strip() or "5")
                except ValueError:
                    n_results = 5

                self.console.print(
                    "[cyan]Minimum similarity threshold (0-1, default: 0.7):[/cyan]"
                )
                try:
                    min_similarity = float(input().strip() or "0.7")
                except ValueError:
                    min_similarity = 0.7

                # Get available models
                available_models = self.processor.get_available_models()
                self.console.print(
                    f"\nAvailable models: {', '.join(available_models)}"
                )
                self.console.print(
                    "[cyan]Specific model to use (press Enter for all):[/cyan]"
                )
                specific_model = input().strip()
                if (
                    specific_model
                    and specific_model not in available_models
                ):
                    self.console.print(
                        "[red]Invalid model selected. Using all models.[/red]"
                    )
                    specific_model = None

                # Execute query
                self.console.print("\n[cyan]Searching...[/cyan]")
                results = await self.processor.query(
                    query,
                    n_results=n_results,
                    min_similarity=min_similarity,
                    specific_model=specific_model,
                )

                # Display results
                if not results:
                    self.console.print("[yellow]No results found.[/yellow]")
                    continue

                for i, result in enumerate(results, 1):
                    similarity = (1 - result.distance) * 100
                    self.console.print(f"\n[green]Result {i}:[/green]")
                    self.console.print(f"Model: {result.model}")
                    self.console.print(f"Document: {result.document_id}")
                    self.console.print(f"Similarity: {similarity:.2f}%")
                    self.console.print(
                        "Content preview:", result.content[:200] + "..."
                    )

        except KeyboardInterrupt:
            self.console.print(
                "\n[yellow]Query session terminated.[/yellow]"
            )
        finally:
            await self.processor.cleanup()


async def get_query_interface(db_path: str) -> QueryInterface:
    """Factory function to create a QueryInterface instance."""
    return QueryInterface(db_path)
