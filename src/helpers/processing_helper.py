"""
Helper module for all helpers
"""

import glob
import json
import logging

from pathlib import Path
from typing import Dict, List, Union

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
)


from ..services.document.pdf_processor import VisionPDFProcessor
from ..services.document.vector_processor import VectorProcessor

logger = logging.getLogger(__name__)
console = Console()


class ProcessingHelper:
    """Helper class to manage document processing operations."""

    def __init__(self, base_path: str = "./src"):
        self.base_path = Path(base_path)
        self.files_path = self.base_path / "files_to_process"
        self.db_path = self.base_path / "data/vector_db"
        self.json_path = self.base_path / "data/json"

        self._setup_directories()

    def _setup_directories(self) -> None:
        """Ensure all required directories exist."""
        for path in [self.files_path, self.db_path, self.json_path]:
            path.mkdir(parents=True, exist_ok=True)

    def print_status(self, status: tuple) -> None:
        """Print processing status with color coding."""
        file_path, status_code, _ = status
        status_map = {
            1: ("success", "green"),
            2: ("skipped", "yellow"),
            3: ("error", "red"),
        }
        status_text, color = status_map.get(
            status_code, ("unknown", "white")
        )
        console.print(f"[{color}]{file_path}: {status_text}[/{color}]")

    async def process_pdfs_to_json(self) -> Dict[str, Union[str, int]]:
        """Process PDF files to JSON format with progress tracking."""
        pdf_files = list(self.files_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDFs found in %s", self.files_path)
            return {"status": "no_files", "processed": 0, "total": 0}

        processor = VisionPDFProcessor()
        results = []

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            pdf_task = progress.add_task(
                "[cyan]Processing PDFs...", total=len(pdf_files)
            )

            try:
                for pdf in pdf_files:
                    status = await processor.process_single_pdf(pdf)
                    self.print_status(status)
                    results.append(status)
                    progress.advance(pdf_task)

            except Exception as e:
                logger.error("Error processing PDFs: %s", e)
                return {"status": "error", "error": str(e)}

        successful = sum(1 for status in results if status[1] == 1)
        skipped = sum(1 for status in results if status[1] == 2)
        errors = sum(1 for status in results if status[1] == 3)

        return {
            "status": "completed",
            "total": len(pdf_files),
            "successful": successful,
            "skipped": skipped,
            "errors": errors,
        }

    def find_json_files(self) -> List[str]:
        """Find all JSON files in the json directory."""
        return glob.glob(
            str(self.json_path / "**" / "*.json"), recursive=True
        )

    def convert_json_to_text(self, json_path: str) -> str:
        """Convert JSON file content to formatted text."""
        # logger.info(f"Converting to text {json_path}")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.dumps(json.load(f), indent=2)
        except Exception as e:
            logger.error("Error converting JSON to text: %s", e)
            return ""

    async def process_json_to_vector_db(
        self, emb_models
    ) -> Dict[str, Union[str, int]]:
        """Process JSON files to vector database."""

        json_paths = self.find_json_files()

        if not json_paths:
            logger.warning("No JSON files found to process")
            return {"status": "no_files", "processed": 0}

        vector_processor = VectorProcessor(str(self.db_path), emb_models)
        texts = self.process_json_paths(json_paths)

        if not texts:
            return {"status": "no_valid_content", "processed": 0}

        try:
            processed_text = await vector_processor.process_text(texts)
            return {
                "status": "success" if processed_text else "error",
                "processed": len(texts) if processed_text else 0,
                "total_files": len(json_paths),
                "processed_text": processed_text,
            }
        except Exception as e:
            logger.error("Error processing JSON to vector database: %s", e)
            return {"status": "error", "error": str(e)}

    def process_json_paths(self, json_paths):
        """Makes sure we know where to write the json path"""
        texts = []
        for json_path in json_paths:
            text = self.convert_json_to_text(json_path)
            if text:
                texts.append((text, Path(json_path).stem))
        return texts
