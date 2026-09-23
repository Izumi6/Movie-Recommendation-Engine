"""
setup_data.py
-------------
Downloads and extracts the MovieLens Latest Small dataset from GroupLens.
The dataset contains ~100K ratings across 9,742 movies from 610 users,
which is ideal for running on a laptop without heavy compute.

Run this script once before launching the app:
    python setup_data.py
"""

import os
import zipfile
import requests

# Where the raw data will live
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATASET_DIR = os.path.join(DATA_DIR, "ml-latest-small")
ZIP_PATH = os.path.join(DATA_DIR, "ml-latest-small.zip")

DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Files we expect inside the zip
EXPECTED_FILES = ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]


def dataset_exists():
    """Check whether all expected CSV files are already on disk."""
    if not os.path.isdir(DATASET_DIR):
        return False
    return all(
        os.path.isfile(os.path.join(DATASET_DIR, f)) for f in EXPECTED_FILES
    )


def download_dataset():
    """Download the MovieLens zip from GroupLens servers."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading MovieLens dataset from {DOWNLOAD_URL} ...")

    response = requests.get(DOWNLOAD_URL, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

    print("\n  Download complete.")


def extract_dataset():
    """Unzip into the data/ directory."""
    print("Extracting dataset ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"  Extracted to {DATASET_DIR}")

    # Clean up the zip to save space
    os.remove(ZIP_PATH)
    print("  Removed zip file.")


def setup():
    """Main entry point — download + extract if needed."""
    if dataset_exists():
        print("Dataset already present. Skipping download.")
        return DATASET_DIR

    download_dataset()
    extract_dataset()

    if not dataset_exists():
        raise FileNotFoundError(
            "Something went wrong — expected CSV files are missing after extraction."
        )

    print("Setup complete. Dataset is ready.")
    return DATASET_DIR


if __name__ == "__main__":
    setup()
