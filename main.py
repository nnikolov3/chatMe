"""
Main entering of the app
"""

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys

from rich.console import Console
from rich.logging import RichHandler

from src.services.chat.chat_processor import EnhancedChatProcessor
from src.services.optimization.resources import ResourceManager
from src.services.query import get_query_interface
from src.helpers.processing_helper import ProcessingHelper

# Initialize rich console for better output formatting
console = Console()

# Configure logging with rich handler for better formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

emb_models = [
    "bge-m3",
    "paraphrase-multilingual",
    "mxbai-embed-large",
    "nomic-embed-text",
    "all-minilm",
    "snowflake-arctic-embed2",
]


class ApplicationManager:
    """Manages the lifecycle and state of the application."""

    def __init__(self):

        self.shutdown_event = asyncio.Event()
        self.helper = ProcessingHelper()

    def setup_signal_handlers(self):
        """Set up handlers for graceful shutdown."""

        def signal_handler():
            console.print(
                "\n[yellow]Initiating graceful shutdown...[/yellow]"
            )
            self.shutdown_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal_handler)

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
            result = await self.helper.process_json_to_vector_db(
                emb_models=emb_models
            )
            if result["status"] == "success":
                console.print(
                    f"""
                    [green]Vector database processing completed
                    - Processed:
                    {result['processed']}/{result['total_files']} files[/green]
                    """
                )
                return True
            else:
                console.print(
                    f"[red]Vector database processing failed: {result}[/red]"
                )
                return False
        except Exception as e:
            logger.exception("Error during vector database processing")
            console.print(
                f"[red]Error processing vector database: {str(e)}[/red]"
            )
            return False

    async def start_query_interface(self) -> bool:
        """Start the interactive query interface with error handling."""
        console.print("[cyan]Starting query interface...[/cyan]")
        try:
            query_interface = await get_query_interface(
                str(self.helper.db_path, emb_models)
            )
            await query_interface.interactive_query()
            return True
        except Exception as e:
            logger.exception("Error in query interface")
            console.print(f"[red]Error in query interface: {str(e)}[/red]")
            return False

    async def start_chat_interface(self) -> bool:
        """Start the enhanced interactive chat interface."""
        console.print("[cyan]Starting enhanced chat interface...[/cyan]")
        try:
            chat = EnhancedChatProcessor(
                "./src/data/vector_db", emb_models=emb_models
            )
            # Needs attent
            await chat.process_chat()

            return True
        except Exception as e:
            logger.exception("Error in chat interface")
            console.print(f"[red]Error in chat interface: {str(e)}[/red]")
            return False

    async def start_api_interface(self) -> bool:
        """Start the API server interface."""
        try:
            subprocess.Popen(
                [
                    "uvicorn",
                    "api:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                ]
            )
            console.print(
                "[green]API server started at 0.0.0.0:8000[/green]"
            )
            return True
        except Exception as e:
            logger.exception("Failed to start API server")
            console.print(
                f"[red]Failed to start API server: {str(e)}[/red]"
            )
            return False


async def main() -> int:
    """Main application entry point with comprehensive error handling.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """

    # Set up Ollama API base URL
    os.environ["OLLAMA_API_BASE_URL"] = "http://127.0.0.1:11434/api"
    os.environ["OLLAMA_NUM_THREADS"] = "8"
    os.environ["OLLAMA_MAX_LOADED"] = "8"
    manager = ResourceManager()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Document processing and query system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--process_pdf",
        action="store_true",
        help="Start processing PDFs to JSON",
    )
    parser.add_argument(
        "-d",
        "--process_db",
        action="store_true",
        help="Start processing JSON to ChromaDB embeddings",
    )
    parser.add_argument(
        "-q",
        "--query",
        action="store_true",
        help="Start interactive query interface",
    )
    parser.add_argument(
        "-c",
        "--chat",
        action="store_true",
        help="Start interactive chat interface",
    )

    parser.add_argument(
        "-a", "--api", action="store_true", help="Start the API server"
    )
    args = parser.parse_args()

    # Initialize application manager
    app_manager = ApplicationManager()
    app_manager.setup_signal_handlers()

    # Use context manager for helper lifecycle

    try:
        if args.process_pdf and not await app_manager.process_pdfs():
            return 1

        if args.process_db and not await app_manager.process_vector_db():
            return 1

        if args.query and not await app_manager.start_query_interface():
            return 1

        if args.chat and not await app_manager.start_chat_interface():
            return 1

        if args.api and not await app_manager.start_api_interface():
            return 1

        if await manager.cleanup():
            return 0
        else:
            sys.exit(1)

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
