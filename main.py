import asyncio
import argparse
import logging
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
import os
import signal
from typing import Optional
from contextlib import asynccontextmanager

from src.helpers.processing_helper import get_helper
from src.services.query.query_processor import get_query_interface

# Initialize rich console for better output formatting
console = Console()

# Configure logging with rich handler for better formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


class ApplicationManager:
    """Manages the lifecycle and state of the application."""

    def __init__(self):
        self.helper = None
        self.shutdown_event = asyncio.Event()

    def _setup_signal_handlers(self):
        """Set up handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            console.print("\n[yellow]Initiating graceful shutdown...[/yellow]")
            self.shutdown_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal_handler)

    @asynccontextmanager
    async def managed_helper(self):
        """Context manager for handling helper lifecycle."""
        try:
            self.helper = await get_helper()
            yield self.helper
        finally:
            if self.helper:
                await self.helper.cleanup_resources()

    async def process_pdfs(self) -> bool:
        """Process PDFs to JSON with enhanced error handling."""
        console.print("[cyan]Processing PDFs to JSON...[/cyan]")
        try:
            result = await self.helper.process_pdfs_to_json()
            if result.get("status") == "completed":
                console.print(
                    f"[green]PDF processing completed:\n"
                    f"- Successful: {result['successful']}/{result['total']}\n"
                    f"- Skipped: {result['skipped']}\n"
                    f"- Errors: {result['errors']}[/green]"
                )
                return True
            else:
                console.print(f"[red]PDF processing failed: {result}[/red]")
                return False
        except Exception as e:
            logger.exception("Error during PDF processing")
            console.print(f"[red]Error processing PDFs: {str(e)}[/red]")
            return False

    async def process_vector_db(self) -> bool:
        """Process JSON to vector database with enhanced error handling."""
        console.print("[cyan]Processing JSON to vector database...[/cyan]")
        try:
            result = await self.helper.process_json_to_vector_db()
            if result["status"] == "success":
                console.print(
                    f"[green]Vector database processing completed:\n"
                    f"- Processed: {result['processed']}/{result['total_files']} files[/green]"
                )
                return True
            else:
                console.print(f"[red]Vector database processing failed: {result}[/red]")
                return False
        except Exception as e:
            logger.exception("Error during vector database processing")
            console.print(f"[red]Error processing vector database: {str(e)}[/red]")
            return False

    async def start_query_interface(self) -> bool:
        """Start the interactive query interface with error handling."""
        console.print("[cyan]Starting query interface...[/cyan]")
        try:
            query_interface = await get_query_interface(str(self.helper.db_path))
            await query_interface.interactive_query()
            return True
        except Exception as e:
            logger.exception("Error in query interface")
            console.print(f"[red]Error in query interface: {str(e)}[/red]")
            return False

    async def export_metrics(self) -> bool:
        """Export resource metrics with error handling."""
        console.print("[cyan]Exporting resource metrics...[/cyan]")
        try:
            if await self.helper.export_resource_metrics():
                console.print("[green]Resource metrics exported successfully[/green]")
                return True
            else:
                console.print("[red]Failed to export resource metrics[/red]")
                return False
        except Exception as e:
            logger.exception("Error exporting metrics")
            console.print(f"[red]Error exporting metrics: {str(e)}[/red]")
            return False


async def main() -> int:
    """Main application entry point with comprehensive error handling.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Set up Ollama API base URL
    os.environ["OLLAMA_API_BASE_URL"] = "http://127.0.0.1:11434/api"

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Document processing and query system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p", "--process", action="store_true", help="Process PDFs to JSON"
    )
    parser.add_argument(
        "-c", "--chroma", action="store_true", help="Process JSON to ChromaDB"
    )
    parser.add_argument(
        "-q", "--query", action="store_true", help="Start interactive query interface"
    )
    parser.add_argument(
        "--export-metrics", action="store_true", help="Export resource metrics"
    )
    args = parser.parse_args()

    # Initialize application manager
    app_manager = ApplicationManager()
    app_manager._setup_signal_handlers()

    # Use context manager for helper lifecycle
    async with app_manager.managed_helper() as helper:
        try:
            # Process PDFs if requested
            if args.process and not await app_manager.process_pdfs():
                return 1

            # Process vector database if requested
            if args.chroma and not await app_manager.process_vector_db():
                return 1

            # Start query interface if requested
            if args.query and not await app_manager.start_query_interface():
                return 1

            # Export metrics if requested
            if args.export_metrics and not await app_manager.export_metrics():
                return 1

            return 0

        except Exception as e:
            logger.exception("Unhandled error in main")
            console.print(f"[red]Unhandled error: {str(e)}[/red]")
            return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Application terminated by user[/yellow]")
        exit(1)
    except Exception as e:
        logger.exception("Critical error")
        console.print(f"[red]Critical error: {str(e)}[/red]")
        exit(1)
