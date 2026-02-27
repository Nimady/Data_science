import pandas as pd

# ----------------------------------------------------------
# This file handles the preprocessing stage of the project.
# It prepares raw textual data for information retrieval.
# The main objective is to create a unified "content" field
# combining title, text, and tags.
# ----------------------------------------------------------


def prepare_data(df):
    """
    Prepare the input DataFrame for retrieval.

    This function:
    1. Replaces missing values (NaN) with empty strings.
    2. Safely extracts title, text, and tags columns.
    3. Converts tag lists into strings when necessary.
    4. Merges all textual fields into a single 'content' column.
    5. Normalizes text by converting everything to lowercase.

    Parameters:
        df (pandas.DataFrame): Input dataset (documents or queries)

    Returns:
        pandas.DataFrame: DataFrame with an additional 'content' column
    """

    # Replace all missing values to avoid concatenation errors
    df = df.fillna("")

    # List that will store the processed text of each row
    contents = []

    # Iterate through each row of the DataFrame
    for index, row in df.iterrows():

        # Safely retrieve text fields (avoid KeyError if column is missing)
        title = row.get("title", "")
        text = row.get("text", "")
        tags = row.get("tags", "")

        # If tags are stored as a list, convert them into a single string
        if isinstance(tags, list):
            tags = " ".join(tags)

        # Merge all textual components into one string
        content = title + " " + text + " " + tags

        # Normalize text to lowercase for consistent retrieval
        content = content.lower()

        # Store processed content
        contents.append(content)

    # Add the new column to the DataFrame
    df["content"] = contents

    return df