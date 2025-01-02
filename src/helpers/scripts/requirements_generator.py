import pkg_resources
import pip
import os
import sys
from pathlib import Path


def find_project_files(directory: str, extension: str = ".py") -> list:
    """Find all Python files in the project directory."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                python_files.append(os.path.join(root, file))
    return python_files


def extract_imports(file_path: str) -> set:
    """Extract all import statements from a Python file."""
    imports = set()
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("import ") or line.startswith("from "):
                # Remove comments if any
                line = line.split("#")[0].strip()
                # Get the main package name
                if line.startswith("from "):
                    package = line.split()[1].split(".")[0]
                else:
                    package = line.split()[1].split(".")[0]
                imports.add(package)
    return imports


def get_package_versions(packages: set) -> dict:
    """Get installed versions of packages."""
    versions = {}
    for package in packages:
        try:
            dist = pkg_resources.get_distribution(package)
            versions[package] = dist.version
        except pkg_resources.DistributionNotFound:
            versions[package] = "Not installed"
    return versions


def generate_requirements():
    """Generate requirements.txt with proper versions."""
    # Get project root directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Find all Python files
    python_files = find_project_files(project_dir)

    # Extract all imports
    all_imports = set()
    for file in python_files:
        file_imports = extract_imports(file)
        all_imports.update(file_imports)

    # Remove standard library modules
    stdlib_modules = set(sys.stdlib_modules)
    third_party_imports = {pkg for pkg in all_imports if pkg not in stdlib_modules}

    # Get versions of installed packages
    package_versions = get_package_versions(third_party_imports)

    # Write requirements.txt
    requirements_path = os.path.join(project_dir, "requirements.txt")
    with open(requirements_path, "w") as f:
        for package, version in sorted(package_versions.items()):
            if version != "Not installed":
                f.write(f"{package}=={version}\n")
            else:
                f.write(f"# {package} - version unknown\n")


if __name__ == "__main__":
    generate_requirements()
    print("Requirements.txt has been generated successfully!")
