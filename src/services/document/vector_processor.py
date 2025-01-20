"""
Mostly processing embeddings
"""

import asyncio
import hashlib
import json
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import uuid

import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)


class VectorProcessor:
    """Enhanced vector database operations
    handler with improved document processing."""

    def __init__(self, persist_dir: str, models: list):
        """Initialize the vector processor with advanced configuration."""
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.models = models
        # Initialize thread pool with configurable size
        self.executor = ThreadPoolExecutor(
            max_workers=multiprocessing.cpu_count()
        )

        # Initialize ChromaDB with optimized settings
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True,
            ),
        )

    def _extract_searchable_content(self, text: str) -> str:
        """Extract and clean searchable content from document text.

        Args:
            text: Raw document text that might be JSON or plain text

        Returns:
            Processed and cleaned text ready for embedding
        """
        try:
            # Try to parse as JSON first
            content = json.loads(text)
            extracted_text = [
                content["content"]["ocr"],
                content["content"]["pdf_text"],
            ]

            processed_text = " ".join(extracted_text)

        except json.JSONDecodeError:
            # Handle as plain text if not JSON
            processed_text = text

        return processed_text

    async def process_text(self, texts: List[tuple]):
        """Process text into vector embeddings
        with enhanced content handling."""
        if not texts:
            logger.error("Empty list provided for text processing")
            return False

        all_embeddings = []
        for text, file_name in texts:
            processed_text = self._extract_searchable_content(text)
            tasks = [
                self._embed_text(processed_text, model, file_name)
                for model in self.models
            ]

            embeddings = await asyncio.gather(
                *tasks, return_exceptions=True
            )

            for result in embeddings:
                if isinstance(
                    result, dict
                ):  # Check if result is a successful embedding
                    all_embeddings.append(result)
                else:
                    # Log errors from failed embeddings
                    logger.error(
                        f"Embedding failed: {result}"
                        if result
                        else "Embedding returned None"
                    )

        # Combine all embeddings into one coherent text
        combined_text = " ".join(
            [emb.get("document", "") for emb in all_embeddings if emb]
        )

        return combined_text

    async def _embed_text(
        self, text: str, model: str, file_name: str
    ) -> Optional[Dict]:
        """Helper method to embed text for a specific model."""
        try:
            emb = OllamaEmbeddings(model=model)
            embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: emb.embed_documents([text])
            )
            embd_id = str(uuid.uuid4())
            text_hash = hashlib.md5(text.encode()).hexdigest()
            timestamp = str(
                datetime.now().isoformat(timespec="microseconds")
            )

            collection = self.client.get_or_create_collection(
                name=model,
                metadata={
                    "hnsw:space": "cosine",
                    "timestamp": timestamp,
                    "model": model,
                    "id": embd_id,
                    "hash": text_hash,
                },
            )

            # Check if similar content already exists
            existing = collection.query(
                query_embeddings=embedding,
                n_results=1,
                include=["metadatas"],
            )

            if existing["metadatas"] and existing["metadatas"][0]:
                if existing["metadatas"][0][0].get("hash") == text_hash:
                    return {"document": text, "model": model}

            collection.add(
                documents=[text],
                embeddings=embedding,
                ids=[embd_id],
                metadatas=[
                    {
                        "id": embd_id,
                        "json_path": file_name,
                        "timestamp": timestamp,
                        "model": model,
                        "original_text": text,
                        "processed": True,
                        "hash": text_hash,
                    }
                ],
            )
            return {"document": text, "model": model}

        except Exception as e:
            logger.error(
                "Error processing embedding for model %s , %s ", model, e
            )
            return None

    async def cleanup(self):
        """Cleanup resources and shutdown properly."""
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error("Error during cleanup: %s", e)
