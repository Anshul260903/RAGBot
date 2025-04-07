# RAGBot – Lightweight Hybrid RAG System
# No heavy downloads needed

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# ---------------------------
# 1. Knowledge Base
# ---------------------------
docs = {
    "1": "RAG combines retrieval with large language models to improve accuracy in QA systems.",
    "2": "Graph search captures relationships between entities like people, places, and organizations.",
    "3": "Keyword search is fast but may miss semantic meaning in text.",
    "4": "Semantic search uses embeddings or similarity measures to find related passages.",
    "5": "Real-time search requires handling freshness and dynamic updates in the index."
}

# ---------------------------
# 2. Keyword Search
# ---------------------------
def keyword_search(query, docs):
    return [(doc_id, text) for doc_id, text in docs.items() if query.lower() in text.lower()]

# ---------------------------
# 3. Semantic Search (TF-IDF)
# ---------------------------
vectorizer = TfidfVectorizer()
doc_ids = list(docs.keys())
doc_texts = list(docs.values())
tfidf_matrix = vectorizer.fit_transform(doc_texts)

def semantic_search(query, top_k=2):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_ids = sims.argsort()[::-1][:top_k]
    return [(doc_ids[i], doc_texts[i]) for i in top_ids]

# ---------------------------
# 4. Graph-based Search
# ---------------------------
G = nx.Graph()
G.add_edges_from([
    ("RAG", "retrieval"),
    ("RAG", "LLM"),
    ("Graph search", "entities"),
    ("Semantic search", "similarity"),
    ("real-time", "freshness")
])

def graph_search(query):
    results = []
    if query in G.nodes:
        neighbors = list(G.neighbors(query))
        results = [(query, n) for n in neighbors]
    return results

# ---------------------------
# 5. Dummy LLM
# ---------------------------
def generate_answer(query, retrieved):
    context_parts = []
    for mode, res in retrieved.items():
        for r in res:
            context_parts.append(r[1] if isinstance(r, tuple) else str(r))
    context = " ".join(context_parts) if context_parts else "No context found."
    return f"Answer (rule-based): Based on context, '{query}' relates to -> {context[:200]}..."

# ---------------------------
# 6. Hybrid Retrieval + QA
# ---------------------------
def hybrid_rag(query):
    retrieved = {
        "keyword": keyword_search(query, docs),
        "semantic": semantic_search(query),
        "graph": graph_search(query)
    }
    answer = generate_answer(query, retrieved)
    return retrieved, answer

# ---------------------------
# 7. Demo Run
# ---------------------------
if __name__ == "__main__":
    query = "search"
    retrieved, answer = hybrid_rag(query)

    print(f"\n=== Query: {query} ===")

    for mode, res in retrieved.items():
        print(f"\n{mode.upper()} RESULTS:")
        for r in res:
            print(" -", r)

    print("\n=== LLM Answer ===")
    print(answer)
