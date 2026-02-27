import json
import pandas as pd
import os
from processing_data import *

# ----------------------------------------------------------
# This file is responsible for loading the dataset.
# It dynamically detects all JSON files in the given folder,
# constructs their paths, loads them into pandas DataFrames,
# and returns them as a dictionary.
# ----------------------------------------------------------


def load_data(data_path):
    """
    Scan the provided directory and build a dictionary
    mapping each JSON filename (without extension)
    to its full file path.
    """

    paths = {}

    # Iterate through all files in the directory
    for filename in os.listdir(data_path):

        # Keep only JSON files
        if filename.endswith(".json"):

            # Construct a portable full path (OS-independent)
            full_path = os.path.join(data_path, filename)

            # Remove the .json extension to use as dictionary key
            key_name = filename.replace(".json", "")

            paths[key_name] = full_path

    return paths


def load_data_frames(paths):
    """
    Load each JSON file into a pandas DataFrame.

    Parameters:
        paths (dict): dictionary mapping file names to file paths

    Returns:
        dict: dictionary mapping file names to pandas DataFrames
    """

    data_frames = {}

    for key, path in paths.items():
        # Read JSON file into a DataFrame
        df = pd.read_json(path)
        data_frames[key] = df

    return data_frames


def load_project_data(data_path):
    """
    High-level function that:
    1. Creates file paths for all JSON files
    2. Loads them into DataFrames
    3. Returns a dictionary of DataFrames
    """

    paths = load_data(data_path)
    data_frames = load_data_frames(paths)

    return data_frames


# ----------------------------------------------------------
# Execution block (only runs when this file is executed
# directly, not when imported as a module)
# ----------------------------------------------------------
if __name__ == "__main__":

    # Load all project data
    data = load_project_data("data")

    # Basic dataset inspection
    print(data.keys())  # Available datasets
    print(data["docs"].head(5))  # Preview first documents
    print(len(data["docs"]))  # Number of documents
    print(len(data["queries_train"]))  # Number of training queries
    print(data["docs"].shape)  # Dataset dimensions
    print(data["docs"].columns)  # Column names

    # Apply preprocessing to create the "content" column
    docs_df = prepare_data(data["docs"])
    queries_train_df = prepare_data(data["queries_train"])

    print(queries_train_df)