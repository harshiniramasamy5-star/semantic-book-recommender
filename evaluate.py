"""
evaluate.py - Precision@K and Recall@K for the book recommender.
Run:  python evaluate.py
"""

# ---- STEP 1: your test set (edit these with real titles from your dataset) ----
TEST_SET = [
    {
        "query": "a redemption story set against war",
        "relevant_titles": ["The Kite Runner", "All the Light We Cannot See", "Atonement"],
    },
    {
        "query": "a cozy magical story about friendship",
        "relevant_titles": ["Harry Potter and the Sorcerer's Stone", "The House in the Cerulean Sea"],
    },
    {
        "query": "a dark psychological thriller with a twist",
        "relevant_titles": ["Gone Girl", "The Silent Patient", "Sharp Objects"],
    },
]

# ---- STEP 2: the metrics ----
def precision_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for t in top_k if t in relevant_set)
    return hits / k

def recall_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for t in top_k if t in relevant_set)
    return hits / len(relevant_set) if relevant_set else 0.0

# ---- STEP 3: connect to YOUR recommender (replace the placeholder body) ----
def get_recommendations(query, k=10):
    # REPLACE THIS with a call to your real function, e.g.:
    #   from main import retrieve_semantic_recommendations
    #   df = retrieve_semantic_recommendations(query, top_k=k)
    #   return df["title"].tolist()
    fake = {
        "a redemption story set against war":
            ["The Kite Runner", "War and Peace", "Atonement", "Dune", "1984"],
        "a cozy magical story about friendship":
            ["The House in the Cerulean Sea", "Dune", "Harry Potter and the Sorcerer's Stone"],
        "a dark psychological thriller with a twist":
            ["Gone Girl", "The Silent Patient", "Pride and Prejudice"],
    }
    return fake.get(query, [])[:k]

# ---- STEP 4: run it ----
def run_evaluation(k=5):
    precisions, recalls = [], []
    print(f"\n=== Evaluating at K={k} ===\n")
    for case in TEST_SET:
        q, relevant = case["query"], case["relevant_titles"]
        rec = get_recommendations(q, k=k)
        p = precision_at_k(rec, relevant, k)
        r = recall_at_k(rec, relevant, k)
        precisions.append(p); recalls.append(r)
        print(f"Query: {q!r}")
        print(f"  Recommended: {rec[:k]}")
        print(f"  Relevant   : {relevant}")
        print(f"  P@{k}={p:.2f}  R@{k}={r:.2f}\n")
    print("=== OVERALL ===")
    print(f"Mean Precision@{k}: {sum(precisions)/len(precisions):.3f}")
    print(f"Mean Recall@{k}   : {sum(recalls)/len(recalls):.3f}")

if __name__ == "__main__":
    run_evaluation(k=5)
