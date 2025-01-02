#!/usr/bin/env python
import os
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import re

# Configure logging for the cleanup script itself
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LogCleaner:
    """A class to manage log file cleanup operations with safety measures."""

    def __init__(self, log_dir: str, archive_dir: str = None):
        """Initialize the log cleaner with directory paths and default settings.

        Args:
            log_dir: Directory containing the logs to clean
            archive_dir: Optional directory for archiving logs before deletion
        """
        self.log_dir = Path(log_dir)
        self.archive_dir = Path(archive_dir) if archive_dir else None

        # Ensure directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.archive_dir:
            self.archive_dir.mkdir(parents=True, exist_ok=True)

    def parse_log_date(self, filename: str) -> datetime:
        """Extract date from log filename using regular expression.

        Args:
            filename: Name of the log file

        Returns:
            datetime object representing the log file's date
        """
        # Extract date pattern (YYYYMMDD_HHMMSS) from filename
        match = re.search(r"(\d{8}_\d{6})", filename)
        if match:
            date_str = match.group(1)
            try:
                return datetime.strptime(date_str, "%Y%m%d_%H%M%S")
            except ValueError:
                logger.warning(f"Could not parse date from filename: {filename}")
        return datetime.fromtimestamp(os.path.getctime(str(self.log_dir / filename)))

    def list_logs(self, days_old: int = None) -> list:
        """List all log files, optionally filtered by age.

        Args:
            days_old: Optional number of days to filter logs by age

        Returns:
            List of log files matching the criteria
        """
        log_files = []
        for file in self.log_dir.glob("*.log"):
            if days_old is not None:
                file_date = self.parse_log_date(file.name)
                if datetime.now() - file_date < timedelta(days=days_old):
                    continue
            log_files.append(file)
        return log_files

    def archive_logs(self, logs_to_archive: list) -> bool:
        """Archive specified log files before deletion.

        Args:
            logs_to_archive: List of log files to archive

        Returns:
            bool: True if archiving was successful
        """
        if not self.archive_dir or not logs_to_archive:
            return False

        try:
            archive_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"logs_archive_{archive_date}"
            archive_path = self.archive_dir / archive_name

            # Create new archive directory
            archive_path.mkdir(exist_ok=True)

            # Copy logs to archive
            for log_file in logs_to_archive:
                shutil.copy2(log_file, archive_path)

            # Create a compressed archive
            shutil.make_archive(str(archive_path), "zip", str(archive_path))

            # Remove the temporary directory
            shutil.rmtree(archive_path)

            logger.info(f"Successfully archived logs to {archive_path}.zip")
            return True

        except Exception as e:
            logger.error(f"Failed to archive logs: {e}")
            return False

    def cleanup_logs(
        self, days_old: int = 7, keep_last: int = 5, archive: bool = True
    ) -> tuple:
        """Clean up old log files with configurable options.

        Args:
            days_old: Delete logs older than this many days
            keep_last: Minimum number of most recent logs to keep
            archive: Whether to archive logs before deletion

        Returns:
            tuple: (number of deleted files, number of archived files)
        """
        # List all log files older than specified days
        old_logs = self.list_logs(days_old)

        # Sort logs by date, newest first
        old_logs.sort(key=lambda x: self.parse_log_date(x.name), reverse=True)

        # Keep the specified number of most recent logs
        logs_to_delete = old_logs[keep_last:]

        if not logs_to_delete:
            logger.info("No logs to clean up")
            return 0, 0

        # Archive logs if requested
        archived_count = 0
        if archive and self.archive_dir:
            if self.archive_logs(logs_to_delete):
                archived_count = len(logs_to_delete)

        # Delete logs
        deleted_count = 0
        for log_file in logs_to_delete:
            try:
                log_file.unlink()
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete {log_file}: {e}")

        logger.info(f"Cleaned up {deleted_count} log files")
        if archived_count:
            logger.info(f"Archived {archived_count} log files")

        return deleted_count, archived_count


def main():
    """Main function to handle command line arguments and execute cleanup."""
    parser = argparse.ArgumentParser(description="Clean up old log files")
    parser.add_argument(
        "--log-dir", default="./src/log", help="Directory containing log files"
    )
    parser.add_argument(
        "--archive-dir",
        default="./src/log/archive",
        help="Directory for archiving logs",
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Delete logs older than this many days"
    )
    parser.add_argument(
        "--keep-last", type=int, default=5, help="Number of recent logs to keep"
    )
    parser.add_argument(
        "--no-archive", action="store_true", help="Skip archiving logs before deletion"
    )

    args = parser.parse_args()

    cleaner = LogCleaner(args.log_dir, args.archive_dir)
    deleted, archived = cleaner.cleanup_logs(
        days_old=args.days, keep_last=args.keep_last, archive=not args.no_archive
    )

    if deleted == 0:
        print("No logs were deleted")
    else:
        print(f"Cleaned up {deleted} log files")
        if archived:
            print(f"Archived {archived} log files")


if __name__ == "__main__":
    main()
