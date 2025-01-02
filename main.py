import asyncio
import argparse
import logging
from rich.console import Console
import os

from src.helpers.processing_helper import get_helper

# Initialize logging and console
console = Console()
logger = logging.getLogger(__name__)


async def main():
    # Set up Ollama API base URL
    os.environ["OLLAMA_API_BASE_URL"] = "http://127.0.0.1:11434/api"

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process documents into vector database"
    )
    parser.add_argument(
        "-p", "--process", action="store_true", help="Process PDFs to JSON"
    )
    parser.add_argument(
        "-c", "--chroma", action="store_true", help="Process JSON to ChromaDB"
    )
    parser.add_argument(
        "--export-metrics", action="store_true", help="Export resource metrics"
    )
    args = parser.parse_args()

    try:
        # Initialize helper
        helper = await get_helper()

        try:
            if args.process:
                console.print("[cyan]Processing PDFs to JSON...[/cyan]")
                result = await helper.process_pdfs_to_json()
                console.print(f"[green]PDF processing completed: {result}[/green]")

            if args.chroma:
                console.print("[cyan]Processing JSON to vector database...[/cyan]")
                result = await helper.process_json_to_vector_db()
                console.print(
                    f"[green]Vector database processing completed: {result}[/green]"
                )

            if args.export_metrics:
                console.print("[cyan]Exporting resource metrics...[/cyan]")
                if await helper.export_resource_metrics():
                    console.print(
                        "[green]Resource metrics exported successfully[/green]"
                    )
                else:
                    console.print("[red]Failed to export resource metrics[/red]")

        finally:
            # Always cleanup resources
            await helper.cleanup_resources()

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return 1

    return 0


if __name__ == "__main__":
    asyncio.run(main())
