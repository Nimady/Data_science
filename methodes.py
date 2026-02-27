from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
import numpy as np


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
        
        
    
    