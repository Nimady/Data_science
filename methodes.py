from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
import numpy as np
from read_data import *
from processing_data import *
from rank_bm25 import BM25Plus


# We fit the TF-IDF vectorizer only on the documents to learn
# the global vocabulary and IDF values. The queries are then
# transformed using the same vector space to ensure that
# document and query vectors have consistent dimensions,
# allowing correct cosine similarity computation.




# We use the "content" column instead of only "text"
# because it combines title, text, and tags into a single
# unified representation. This allows the retrieval model
# to use all available textual information, not just the body text,
# improving the quality of similarity matching.


def tfidf_retrieve(docs_df,queries_df,top_k=10):
    
    docs_content=docs_df["content"]
    queries_content=queries_df["content"]
    
    vectorizer=TfidfVectorizer()
    # TF-IDF first learns a global vocabulary from the document collection.
    # This vocabulary defines a fixed vector space where each dimension
    # corresponds to a unique word. Queries are then transformed into
    # this same vector space to ensure consistent dimensions, allowing
    # correct cosine similarity computation between queries and documents.
    docs_content_vec=vectorizer.fit_transform(docs_content)
    queries_content_vec=vectorizer.transform(queries_content)   
   
   
    similarity_matrix = cosine_similarity(queries_content_vec, docs_content_vec) 
    
        # For each query, retrieve top_k most similar documents
    results = []
    for i in range(len(queries_df)):

        scores = similarity_matrix[i]

        # Get indices of top_k highest similarity scores
        top_indices = scores.argsort()[-top_k:][::-1]

        # Convert indices to document IDs
        top_doc_ids = docs_df.iloc[top_indices]["id"].tolist()

        # Store results
        results.append({
            "query_id": queries_df.iloc[i]["id"],
            "relevant_docs": top_doc_ids
        })

    return results    



#pour TOI
# This function implements a TF-IDF based retrieval system.
# It first learns the global vocabulary and IDF values from the document collection.
# Then, it transforms both documents and queries into the same vector space.
# Cosine similarity is computed between each query and all documents.
# For every query, the function selects the top_k most similar documents
# and returns their IDs ranked from most to least relevant.


from rank_bm25 import BM25Plus
import numpy as np


def bm25_retrieve(docs_df, queries_df, top_k=10):
    """
    This function implements a BM25 retrieval system.
    It builds a BM25 index from the document collection,
    computes a relevance score for each query-document pair,
    and returns the top_k most relevant document IDs per query.
    """

    # Tokenize documents (BM25 expects list of word lists)
    tokenized_docs = [doc.split() for doc in docs_df["content"]]

    # Build BM25 index (learns term frequencies and IDF)
    bm25 = BM25Plus(tokenized_docs)

    results = []
    # We tokenize all documents once at the beginning because BM25
# builds an index based on the entire document collection.
# This index depends only on the documents and does not change per query.
# Queries, however, are processed one by one, so we tokenize each query
# inside the loop when computing its relevance scores.

    # For each query
    for index, row in queries_df.iterrows():

        query_text = row["content"]

        # Tokenize query
        tokenized_query = query_text.split()

        # Compute BM25 scores against all documents
        doc_scores = bm25.get_scores(tokenized_query)

        # Select top_k highest scores
        top_indices = np.argsort(doc_scores)[-top_k:][::-1]

        # Convert indices to document IDs
        top_doc_ids = docs_df.iloc[top_indices]["id"].tolist()

        results.append({
            "query_id": row["id"],
            "relevant_docs": top_doc_ids
        })

    return results
    
   #pour toi
# This function implements a BM25-based retrieval system.
# It first tokenizes all documents and builds a BM25 index from them.
# This index stores term frequencies and inverse document frequencies (IDF)
# needed to compute relevance scores.
#
# For each query:
# - The query is tokenized into words.
# - A BM25 relevance score is computed between the query and every document.
# - The documents are ranked according to their BM25 score.
# - The top_k most relevant document IDs are returned.
#
# The function outputs a list of dictionaries, where each dictionary
# contains a query_id and the list of its top_k ranked document IDs.
   
    
    

        
        
        
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

    # Run TF-IDF retrieval and print results
    tfidf_results = tfidf_retrieve(docs_df, queries_train_df, top_k=10)
    print(tfidf_results[:2])  # Print first 2 results for inspection
    
    
        # 🔹 Reduce size for testing (optional but recommended)
    small_docs = docs_df.iloc[:5000]       # first 5000 documents
    small_queries = queries_train_df.iloc[:3]  # first 3 queries

    # Run BM25
    results = bm25_retrieve(small_docs, small_queries, top_k=5)

    # Print results
    for res in results:
        print("Query ID:", res["query_id"])
        print("Top Docs:", res["relevant_docs"])
        print("-" * 40)
        
    
    