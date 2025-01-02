import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
import glob
import json
import os

from ..services.document.pdf_processor import VisionPDFProcessor
from ..services.document.vector_processor import VectorProcessor
from ..optimization.resources import ResourceManager

logger = logging.getLogger(__name__)
console = Console()


class ProcessingHelper:
    """Helper class to manage document processing operations."""

    def __init__(self, base_path: str = "./src"):
        self.base_path = Path(base_path)
        self.files_path = self.base_path / "files_to_process"
        self.db_path = self.base_path / "data/vector_db"
        self.json_path = self.base_path / "data/json"
        self.resource_manager = ResourceManager()

        # Ensure all required directories exist
        self._setup_directories()

    def _setup_directories(self):
        """Ensure all required directories exist."""
        for path in [self.files_path, self.db_path, self.json_path]:
            path.mkdir(parents=True, exist_ok=True)

    async def initialize_resources(self):
        """Initialize and start resource monitoring."""
        try:
            await self.resource_manager.start_monitoring()
            logger.info("Resource monitoring started successfully")
        except Exception as e:
            logger.error(f"Failed to initialize resource monitoring: {e}")
            raise

    async def cleanup_resources(self):
        """Cleanup and stop resource monitoring."""
        try:
            await self.resource_manager.cleanup()
            logger.info("Resource cleanup completed")
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")

    def print_status(self, status: tuple) -> None:
        """Print processing status with color coding."""
        file_path, status_code, _ = status
        status_map = {
            1: ("success", "green"),
            2: ("skipped", "yellow"),
            3: ("error", "red"),
        }

        status_text, color = status_map.get(status_code, ("unknown", "white"))
        console.print(f"[{color}]{file_path}: {status_text}[/{color}]")

    async def process_pdfs_to_json(self) -> Dict:
        """Process PDF files to JSON format with progress tracking."""
        pdf_files = list(self.files_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDFs found in {self.files_path}")
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
                logger.error(f"Error processing PDFs: {e}")
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
        pattern = os.path.join(self.json_path, "**", "*.json")
        return glob.glob(pattern, recursive=True)

    def convert_json_to_text(self, json_path: str) -> str:
        """Convert JSON file content to formatted text."""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        except Exception as e:
            logger.error(f"Error converting JSON to text: {e}")
            return ""

    async def process_json_to_vector_db(self) -> Dict:
        """Process JSON files to vector database."""
        try:
            vector_processor = VectorProcessor(str(self.db_path))
            json_paths = self.find_json_files()

            if not json_paths:
                logger.warning("No JSON files found to process")
                return {"status": "no_files", "processed": 0}

            texts = []
            for json_path in json_paths:
                text = self.convert_json_to_text(json_path)
                if text:
                    file_name = Path(json_path).stem
                    texts.append((text, file_name))

            if not texts:
                return {"status": "no_valid_content", "processed": 0}

            success = await vector_processor.process_text(texts)
            return {
                "status": "success" if success else "error",
                "processed": len(texts) if success else 0,
                "total_files": len(json_paths),
            }

        except Exception as e:
            logger.error(f"Error processing JSON to vector database: {e}")
            return {"status": "error", "error": str(e)}

    async def export_resource_metrics(self, filepath: Optional[str] = None) -> bool:
        """Export resource metrics to a file."""
        if filepath is None:
            filepath = self.base_path / "data/resource_metrics.json"

        try:
            await self.resource_manager.export_metrics(str(filepath))
            return True
        except Exception as e:
            logger.error(f"Failed to export resource metrics: {e}")
            return False


async def get_helper(base_path: str = "./src") -> ProcessingHelper:
    """Factory function to create and initialize a ProcessingHelper instance."""
    helper = ProcessingHelper(base_path)
    await helper.initialize_resources()
    return helper
