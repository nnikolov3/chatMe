import os
import subprocess


def format_with_black(start_dir):
    """
    Recursively formats Python files in the specified directory using the Black formatter.

    :param start_dir: The root directory to start the search from.
    """
    for root, _, files in os.walk(start_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    # Run Black with settings optimized for readability
                    subprocess.run(
                        [
                            "black",
                            "--line-length",
                            "76",  # Default line length for readability
                            "--color",
                            "-W",
                            "5",
                            file_path,
                        ],
                        check=True,
                    )
                    print(f"Formatted: {file_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error formatting {file_path}: {e}")


if __name__ == "__main__":
    # Change this to the directory you want to start from
    directory = input("Enter the directory to format recursively: ")

    if os.path.isdir(directory):
        format_with_black(directory)
    else:
        print("The specified path is not a directory.")
