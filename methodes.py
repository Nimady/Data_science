import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Plus

from read_data import *
from processing_data import *

# ----------------------------------------------------------
# This file implements Phase 1 retrieval baselines:
# - TF-IDF + cosine similarity
# - BM25 (BM25Plus variant)
#
# This version is optimized for technical datasets containing:
#   asp.net, c#, c++, node.js, tikz-pgf, \href, etc.
#
# IMPORTANT:
# Tokenization must preserve technical symbols:
#   . - _ + # \
# ----------------------------------------------------------


# ----------------------------------------------------------
# Shared tokenizer (consistent with processing_data.py)
# ----------------------------------------------------------

# Keep letters, digits and important technical symbols
TOKEN_PATTERN = re.compile(r"[a-z0-9\-\._\+#\\]+")


def tokenize(text: str):
    """
    Tokenize text while preserving technical tokens such as:
    asp.net, c#, node.js, tikz-pgf, \href

    Parameters:
        text (str)

    Returns:
        list[str]
    """
    return TOKEN_PATTERN.findall(str(text).lower())


# ==========================================================
# TF-IDF Retrieval
# ==========================================================

def tfidf_retrieve(docs_df, queries_df, top_k=10):
    """
    Perform document retrieval using TF-IDF + cosine similarity.

    Steps:
    1. Learn vocabulary and IDF from documents.
    2. Transform documents and queries into TF-IDF vectors.
    3. Compute cosine similarity.
    4. Return top_k most relevant documents per query.

    Parameters:
        docs_df (DataFrame): ['id', 'content']
        queries_df (DataFrame): ['id', 'content']
        top_k (int)

    Returns:
        (indices, scores, results)
    """

    docs_content = docs_df["content"].astype(str)
    queries_content = queries_df["content"].astype(str)

    # IMPORTANT:
    # Use custom tokenizer that preserves technical tokens.
    # Do NOT use stopwords filtering for this dataset.
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        lowercase=False,
        token_pattern=None,
        ngram_range=(1, 2),
        norm="l2"
    )

    # Learn vocabulary from documents
    docs_vec = vectorizer.fit_transform(docs_content)

    # Transform queries into same space
    queries_vec = vectorizer.transform(queries_content)

    # Cosine similarity matrix
    similarity_matrix = cosine_similarity(queries_vec, docs_vec)

    doc_ids = docs_df["id"].to_numpy()
    query_ids = queries_df["id"].to_numpy()

    topk_indices = []
    topk_scores = []
    results = []

    for i, query_id in enumerate(query_ids):

        scores = similarity_matrix[i]

        # Efficient top-k selection
        if top_k >= len(scores):
            indices = np.argsort(scores)[::-1]
        else:
            indices = np.argpartition(scores, -top_k)[-top_k:]
            indices = indices[np.argsort(scores[indices])[::-1]]

        selected_scores = scores[indices]
        selected_doc_ids = doc_ids[indices].tolist()

        topk_indices.append(indices.tolist())
        topk_scores.append(selected_scores.tolist())

        results.append({
            "query_id": query_id,
            "relevant_docs": selected_doc_ids,
            "scores": selected_scores.tolist()
        })

    return topk_indices, topk_scores, results


# ==========================================================
# BM25 Retrieval
# ==========================================================

def bm25_retrieve(docs_df, queries_df, top_k=10,
                  k1=1.5, b=0.75, delta=1.0):
    """
    Perform document retrieval using BM25Plus.

    Parameters:
        docs_df (DataFrame): ['id', 'content']
        queries_df (DataFrame): ['id', 'content']
        top_k (int)
        k1, b, delta: BM25 parameters

    Returns:
        (indices, scores, results)
    """

    # Tokenize documents
    tokenized_docs = [
        tokenize(doc)
        for doc in docs_df["content"].astype(str)
    ]

    # Build BM25 index
    bm25 = BM25Plus(
        tokenized_docs,
        k1=k1,
        b=b,
        delta=delta
    )

    doc_ids = docs_df["id"].to_numpy()
    query_ids = queries_df["id"].to_numpy()

    topk_indices = []
    topk_scores = []
    results = []

    for query_id, query in zip(query_ids,
                               queries_df["content"].astype(str)):

        tokenized_query = tokenize(query)

        scores = bm25.get_scores(tokenized_query)

        # Efficient top-k
        if top_k >= len(scores):
            indices = np.argsort(scores)[::-1]
        else:
            indices = np.argpartition(scores, -top_k)[-top_k:]
            indices = indices[np.argsort(scores[indices])[::-1]]

        selected_scores = scores[indices]
        selected_doc_ids = doc_ids[indices].tolist()

        topk_indices.append(indices.tolist())
        topk_scores.append(selected_scores.tolist())

        results.append({
            "query_id": query_id,
            "relevant_docs": selected_doc_ids,
            "scores": selected_scores.tolist()
        })

    return topk_indices, topk_scores, results


# ==========================================================
# Test block
# ==========================================================

if __name__ == "__main__":

    # Load data
    data = load_project_data("data")

    docs_df = prepare_data(data["docs"])
    queries_df = prepare_data(data["queries_train"])

    print("Docs:", len(docs_df))
    print("Queries:", len(queries_df))

    # TF-IDF test
    _, _, tfidf_results = tfidf_retrieve(
        docs_df,
        queries_df,
        top_k=5
    )

    print("\nTF-IDF sample:")
    print(tfidf_results[:2])

    # BM25 test
    _, _, bm25_results = bm25_retrieve(
        docs_df,
        queries_df,
        top_k=5
    )

    print("\nBM25 sample:")
    print(bm25_results[:2])