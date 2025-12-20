# setup_project.py
import os
from pathlib import Path

# Create directory structure
dirs = [
    "src",
    "examples", 
    "tests",
    "data/female",
    "data/male",
    "artifacts",
    "feedback_data",
    "logs"
]

for dir_path in dirs:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created {dir_path}/")

# Create __init__.py files
init_files = [
    "src/__init__.py",
    "tests/__init__.py"
]

for init_file in init_files:
    Path(init_file).touch()
    print(f"✓ Created {init_file}")

print("\n✓ Project structure created successfully!")
print("\nNow you can run:")
print("  python examples/train_model.py")