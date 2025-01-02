import os
import argparse
from datetime import datetime


def generate_repo_report(directory, output_file):
    """
    Generate a report containing the tree structure and Python file contents of a directory.

    Args:
        directory (str): The root directory to process.
        output_file (str): The output text file to save the report.
    """

    def get_file_metadata(file_path):
        """Retrieve metadata for a file."""
        try:
            stats = os.stat(file_path)
            size = stats.st_size  # File size in bytes
            modified_time = datetime.fromtimestamp(stats.st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            permissions = oct(stats.st_mode)[-3:]  # Permissions in octal
            return size, modified_time, permissions
        except Exception as e:
            return None, None, f"Error retrieving metadata: {e}"

    def count_lines(file_path):
        """Count total, blank, and comment lines in a file."""
        total = blank = comments = 0
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    total += 1
                    stripped = line.strip()
                    if not stripped:
                        blank += 1
                    elif stripped.startswith("#"):
                        comments += 1
        except Exception as e:
            return total, blank, comments, f"Error counting lines: {e}"
        return total, blank, comments, None

    with open(output_file, "w", encoding="utf-8") as output:
        output.write("Repository Tree:\n")
        output.write("===================\n")

        # Generate the recursive tree structure first
        for root, dirs, files in os.walk(directory):
            if ".git" in dirs:
                dirs.remove(".git")  # Exclude .git directory

            indent_level = root.replace(directory, "").count(os.sep)
            output.write(f"{'    ' * indent_level}{os.path.basename(root)}/\n")

            for file in files:
                if file.endswith(".py"):
                    output.write(f"{'    ' * (indent_level + 1)}{file}\n")

        output.write("\nDetailed File Information:\n")
        output.write("===========================\n")

        # Traverse the directory again for detailed file information
        for root, dirs, files in os.walk(directory):
            if ".git" in dirs:
                dirs.remove(".git")  # Exclude .git directory

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)

                    # Gather metadata
                    size, modified_time, permissions = get_file_metadata(file_path)
                    total, blank, comments, error = count_lines(file_path)

                    # Write detailed file information
                    output.write(f"\n### FILE: {file} ###\n")
                    output.write(f"### DIRECTORY: {root} ###\n")
                    output.write(
                        f"### METADATA: Size={size} bytes, Modified={modified_time}, Permissions={permissions} ###\n"
                    )
                    output.write(
                        f"### LINE COUNTS: Total={total}, Blank={blank}, Comments={comments} ###\n"
                    )
                    output.write(f"{'-' * 40}\n")

                    # Append the content of the file
                    try:
                        with open(file_path, "r", encoding="utf-8") as file_content:
                            output.write(file_content.read())
                        output.write(f"\n{'=' * 80}\n")
                    except Exception as e:
                        output.write(f"### ERROR READING FILE: {e} ###\n")
                        output.write(f"\n{'=' * 80}\n")

    print(
        f"Tree structure and Python file contents with metadata saved to {output_file}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a comprehensive repo report with metadata."
    )
    parser.add_argument(
        "-d", "--directory", required=True, help="The root directory to process."
    )
    parser.add_argument("-o", "--output", required=True, help="The output text file.")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a valid directory.")
    else:
        generate_repo_report(args.directory, args.output)
