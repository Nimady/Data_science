import re
import pandas as pd

# ----------------------------------------------------------
# This file handles the preprocessing stage of the project.
# It prepares raw textual data for information retrieval.
# The main objective is to create a unified "content" field
# combining title, text, and tags.
#
# IMPORTANT (dataset-specific):
# This dataset contains many technical tokens where punctuation
# is meaningful (e.g., "asp.net-mvc", "c#", "c++", "node.js",
# "\href", "tikz-pgf"). If we remove all punctuation aggressively,
# we destroy these tokens and retrieval quality drops sharply.
#
# Therefore, we keep a safe set of characters commonly useful in
# technical IR:
#   - hyphen   "-"   (tags like tikz-pgf)
#   - dot      "."   (node.js, asp.net)
#   - underscore "_" (snake_case)
#   - plus     "+"   (c++)
#   - hash     "#"   (c#)
#   - backslash "\"  (LaTeX commands like \href, \begin)
# Everything else is replaced by spaces, and whitespace is normalized.
# ----------------------------------------------------------


# Precompiled regex for speed and consistency:
# Keep: letters, digits, whitespace, and the technical chars: - . _ + # \
_ALLOWED_CHARS_RE = re.compile(r"[^a-z0-9\s\-\._\+#\\]+")

# Whitespace normalization (collapse multiple spaces/tabs/newlines)
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Clean and normalize raw text for retrieval.

    This function:
    - Lowercases text for consistent matching.
    - Preserves technical characters that are meaningful in this dataset:
      - . - _ + # \\
    - Replaces other characters with spaces.
    - Normalizes whitespace.

    Parameters:
        text (str): Raw text

    Returns:
        str: Cleaned and normalized text
    """
    # Ensure it's a string
    text = "" if text is None else str(text)

    # Normalize to lowercase for consistent retrieval
    text = text.lower()

    # Replace disallowed characters with spaces (do NOT remove allowed technical chars)
    text = _ALLOWED_CHARS_RE.sub(" ", text)

    # Collapse repeated whitespace
    text = _WS_RE.sub(" ", text).strip()

    return text


def prepare_data(df: pd.DataFrame, boost_tags: bool = True, tag_boost_factor: int = 2) -> pd.DataFrame:
    """
    Prepare the input DataFrame for retrieval.

    This function:
    1. Replaces missing values (NaN) with empty strings.
    2. Safely extracts title, text, and tags columns.
    3. Converts tag lists into strings when necessary.
    4. Merges all textual fields into a single 'content' column.
    5. Cleans and normalizes text using clean_text().
    6. Optionally boosts tags by repeating them (helps retrieval on short technical queries).

    Parameters:
        df (pandas.DataFrame): Input dataset (documents or queries)
        boost_tags (bool): If True, tags are repeated to give them more weight
        tag_boost_factor (int): Number of times to repeat the tags text (>= 1)

    Returns:
        pandas.DataFrame: DataFrame with an additional 'content' column
    """

    # Replace all missing values to avoid concatenation errors
    df = df.fillna("")

    # Ensure tag_boost_factor is valid
    if tag_boost_factor < 1:
        tag_boost_factor = 1

    # List that will store the processed text of each row
    contents = []

    # Iterate through each row of the DataFrame
    for _, row in df.iterrows():

        # Safely retrieve text fields (avoid KeyError if column is missing)
        title = row.get("title", "")
        text = row.get("text", "")
        tags = row.get("tags", "")

        # If tags are stored as a list, convert them into a single string
        if isinstance(tags, list):
            tags = " ".join(map(str, tags))
        else:
            tags = str(tags) if tags is not None else ""

        # Merge all textual components into one string
        # Tag boosting: tags are usually highly informative in technical corpora.
        if boost_tags and tags.strip():
            boosted_tags = (" " + tags) * tag_boost_factor
            content = f"{title} {text}{boosted_tags}"
        else:
            content = f"{title} {text} {tags}"

        # Clean and normalize the final content
        content = clean_text(content)

        # Store processed content
        contents.append(content)

    # Add the new column to the DataFrame
    df["content"] = contents

    return df