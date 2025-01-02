from pathlib import Path
from typing import Dict, List, Tuple
import logging
from chromadb.config import Settings
import time
import chromadb
from langchain_ollama import OllamaEmbeddings
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class VectorProcessor:
    def __init__(self, persist_dir: str):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.models = [
            "bge-m3",
            "paraphrase-multilingual",
            "mxbai-embed-large",
            "nomic-embed-text",
        ]
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def process_text(self, texts: List[tuple]) -> bool:
        if not texts:
            logger.error("Empty list provided for text processing")
            return False

        successful_embeddings = []
        failed_embeddings = []

        for mod in self.models:
            for text, file_name in texts:
                try:
                    db_path = "./src/data/vector_db"
                    success = await self.process_embeddings(
                        text, file_name, mod, db_path
                    )

                    if success:
                        successful_embeddings.append((mod, file_name))
                        logger.info(
                            f"Embedding for model {mod} successful for {file_name}"
                        )
                    else:
                        failed_embeddings.append((mod, file_name))
                        logger.warning(
                            f"Embedding failed for model {mod} and file {file_name}"
                        )

                except Exception as e:
                    failed_embeddings.append((mod, file_name))
                    logger.error(
                        f"Error processing embedding for model {mod}: {str(e)}"
                    )

                # Add a small delay between requests to prevent overloading
                await asyncio.sleep(0.5)

        # Log summary
        logger.info(f"Successfully processed embeddings: {len(successful_embeddings)}")
        if failed_embeddings:
            logger.warning(f"Failed embeddings: {failed_embeddings}")

        # Consider the operation successful if at least one embedding worked
        return len(successful_embeddings) > 0

    async def process_embeddings(
        self, text_to_embed: str, file_name: str, mod: str, db_path: str
    ) -> bool:
        try:
            client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(anonymized_telemetry=False),
            )

            # Use ThreadPoolExecutor for CPU-bound embedding operation
            loop = asyncio.get_event_loop()
            emb = OllamaEmbeddings(model=mod)
            embedding = await loop.run_in_executor(
                self.executor, lambda: emb.embed_documents([text_to_embed])
            )

            meta_data = {
                "db_path": db_path,
                "hnsw:space": "cosine",
                "time": time.time(),
                "model": mod,
                "json_path": file_name,
                "title": file_name,
            }

            collection = client.get_or_create_collection(name=mod, metadata=meta_data)
            collection.upsert(
                documents=text_to_embed,
                embeddings=embedding,
                ids=[f"doc_{time.time()}_{file_name}"],
            )
            return True

        except Exception as e:
            logger.error(f"Embedding failed for model {mod}: {e}")
            return False

    async def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
