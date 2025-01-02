import pkg_resources
import subprocess

# Get a list of outdated packages
outdated_packages = subprocess.run(
    ["pip", "list", "--outdated", "--format=freeze"], capture_output=True, text=True
).stdout.splitlines()

# Extract package names
packages = [line.split("==")[0] for line in outdated_packages]

# Upgrade each package
for package in packages:
    subprocess.run(["pip", "install", "--upgrade", package])
