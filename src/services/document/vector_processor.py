from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from chromadb.config import Settings
import time
import chromadb
from langchain_ollama import OllamaEmbeddings
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import json
import re
import multiprocessing
import hashlib
from functools import reduce

logger = logging.getLogger(__name__)


class VectorProcessor:
    """Enhanced vector database operations handler with improved document processing."""

    def __init__(self, persist_dir: str):
        """Initialize the vector processor with advanced configuration.

        Args:
            persist_dir: Directory path for ChromaDB persistence
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Curated list of models optimized for different aspects of text understanding
        self.models = [
            "bge-m3",  # Good for general text understanding
            "paraphrase-multilingual",  # Strong at handling variations in expression
            "mxbai-embed-large",  # Effective for technical content
            "nomic-embed-text",  # Good at maintaining semantic relationships
        ]

        # Initialize thread pool with configurable size
        self.executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())

        # Initialize ChromaDB with optimized settings
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False, allow_reset=True, is_persistent=True
            ),
        )

        # Initialize content extraction patterns
        self._initialize_extraction_patterns()

    def _initialize_extraction_patterns(self):
        """Initialize regex patterns for content extraction and cleaning."""
        self.patterns = {
            "whitespace": re.compile(r"\s+"),
            "special_chars": re.compile(r"[^\w\s\-.,;?!]"),
            "multiple_dots": re.compile(r"\.{2,}"),
        }

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
            extracted_text = []

            # Handle nested JSON structure
            if isinstance(content, dict):
                if "content" in content and "pages" in content["content"]:
                    for page in content["content"]["pages"]:
                        for page_content in page.values():
                            if "ocr_text" in page_content:
                                text = page_content["ocr_text"].split()
                                extracted_text.append(text)
                            if "first_vision_analysis" in page_content:
                                text = page_content["first_vision_analysis"].split()
                                extracted_text.append(text)
                            if "second_vision_analysis" in page_content:
                                if isinstance(
                                    page_content["second_vision_analysis"], list
                                ):

                                    extracted_text.extend(
                                        page_content["second_vision_analysis"]
                                    )
                                else:
                                    extracted_text.append(
                                        str(page_content["second_vision_analysis"])
                                    )

            # Join all extracted text into one call
            flattened = reduce(
                lambda x, y: reduce(
                    lambda a, b: a + b if isinstance(b, list) else a + [b],
                    y if isinstance(y, list) else [y],
                    x,
                ),
                extracted_text,
                [],
            )
            processed_text = " ".join(flattened)

        except json.JSONDecodeError:
            # Handle as plain text if not JSON
            processed_text = text

        return processed_text

    async def process_text(self, texts: List[tuple]) -> bool:
        """Process text into vector embeddings with enhanced content handling.

        Args:
            texts: List of (text, file_name) tuples to process

        Returns:
            bool indicating if at least one embedding was successful
        """
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

        # Log the combined text, using an ellipsis for larger texts
        logger.info(
            f"Combined text for LLM: {combined_text[:100]}{'...' if len(combined_text) > 100 else ''}"
        )

        return bool(all_embeddings)

    async def _embed_text(
        self, text: str, model: str, file_name: str
    ) -> Optional[Dict]:
        """Helper method to embed text for a specific model."""
        try:
            emb = OllamaEmbeddings(model=model)
            embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: emb.embed_documents([text])
            )

            collection = self.client.get_or_create_collection(
                name=model,
                metadata={
                    "hnsw:space": "cosine",
                    "time": time.time(),
                    "model": model,
                },
            )
            # Hash the text content
            text_hash = hashlib.md5(text.encode()).hexdigest()

            # Use this hash in the ID or metadata to ensure content uniqueness
            unique_id = f"doc_{text_hash}_{file_name}_{model}"

            collection.add(
                documents=[text],
                embeddings=embedding,
                ids=[unique_id],
                metadatas=[
                    {
                        "id": unique_id,  # Add the generated ID to the metadata
                        "json_path": file_name,
                        "timestamp": time.time(),
                        "model": model,
                        "original_text": text,
                        "processed": True,
                    }
                ],
            )
            logger.info(f"Successfully processed {file_name} with {model}")
            return {"document": text, "model": model}

        except Exception as e:
            logger.error(f"Error processing embedding for model {model}: {str(e)}")
            return None

    async def cleanup(self):
        """Cleanup resources and shutdown properly."""
        try:
            self.executor.shutdown(wait=True)
            logger.info("Vector processor cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
