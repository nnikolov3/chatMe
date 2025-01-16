from pathlib import Path
import secrets
from typing import Dict, List, Optional
import logging
from chromadb.config import Settings
import chromadb
from langchain_ollama import OllamaEmbeddings
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import hashlib
from datetime import datetime
import random

logger = logging.getLogger(__name__)


class VectorProcessor:
    """Enhanced vector database operations handler with improved document processing."""

    def __init__(self, persist_dir: str, models: list):
        """Initialize the vector processor with advanced configuration.

        Args:
            persist_dir: Directory path for ChromaDB persistence
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Curated list of models optimized for different aspects of text understanding
        self.models = models
        # Initialize thread pool with configurable size
        self.executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())

        # Initialize ChromaDB with optimized settings
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False, allow_reset=True, is_persistent=True
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
            extracted_text = [content["content"]["ocr"], content["content"]["pdf_text"]]

            processed_text = " ".join(extracted_text)

        except json.JSONDecodeError:
            # Handle as plain text if not JSON
            processed_text = text

        return processed_text

    async def process_text(self, texts: List[tuple]):
        """Process text into vector embeddings with enhanced content handling."""
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

            # Use asyncio.gather with return_exceptions=True to handle potential failures gracefully
            embeddings = await asyncio.gather(*tasks, return_exceptions=True)

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

            text_hash = hashlib.md5(text.encode()).hexdigest()
            collection = self.client.get_or_create_collection(
                name=model,
                metadata={
                    "hnsw:space": "cosine",
                    "timestamp": f"{datetime.now().isoformat(timespec="microseconds")}",
                    "date": f"{datetime.date}",
                    "model": model,
                    "id": f"{text_hash}_{random.uniform(0.111111, 99.99999)}",
                    "hash": text_hash,
                    "secret_id": str(secrets.randbelow(10**20)),
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
                    # logger.info(f"Skipping duplicate content with hash {text_hash}")
                    return {"document": text, "model": model}

            text_hash = hashlib.md5(text.encode()).hexdigest()
            unique_id = f"doc_{text_hash}_{file_name}_{model}_{random.uniform(0.111111, 99.99999)}"
            collection = self.client.get_or_create_collection(
                name=model,
                metadata={
                    "hnsw:space": "cosine",
                    "timestamp": f"{datetime.now().isoformat(timespec="microseconds")}",
                    "date": f"{datetime.date}",
                    "model": model,
                    "id": f"{text_hash}_{random.uniform(0.1111111, 999.999990)}",
                    "hash": text_hash,
                    "unique_id": unique_id,
                    "secret_id": str(secrets.randbelow(10**20)),
                },
            )

            collection.add(
                documents=[text],
                embeddings=embedding,
                ids=[unique_id],
                metadatas=[
                    {
                        "id": unique_id,
                        "json_path": file_name,
                        "timestamp": f"{datetime.now().isoformat(timespec="microseconds")}",
                        "date": f"{datetime.date}",
                        "model": model,
                        "original_text": text,
                        "processed": True,
                        "hash": text_hash,
                        "unique_id": unique_id,
                        "secret_id": str(secrets.randbelow(10**20)),
                    }
                ],
            )
            return {"document": text, "model": model}

        except Exception as e:
            logger.error(f"Error processing embedding for model {model}: {str(e)}")
            return None

    async def cleanup(self):
        """Cleanup resources and shutdown properly."""
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
