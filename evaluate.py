"""
evaluate.py - Precision@K and Recall@K for the book recommender.
Wired to the real recommender in GRADIO.py.
Run:  python evaluate.py
"""

from GRADIO import retrieve_semantic_recommendations

# ---- STEP 1: your test set ----
TEST_SET = [
    {
        "query": "a redemption story set against war",
        "relevant_titles": [
            "Catch-22",
            "If I Die in a Combat Zone",
            "The Women's War",
        ],
    },
    {
        "query": "a magical adventure for young readers",
        "relevant_titles": [
            "The Lion, the Witch and the Wardrobe (picture book edition)",
            "The Alchemist",
            "Anansi Boys",
        ],
    },
    {
        "query": "a story about identity and self discovery",
        "relevant_titles": [
            "Identity",
            "The Alchemist",
        ],
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

# ---- STEP 3: call YOUR real recommender ----
def get_recommendations(query, k=10):
    # Pass category="All" and tone="All" so the filter inside the function
    # does not accidentally drop everything.
    df = retrieve_semantic_recommendations(query, category="All", tone="All", final_top_k=k)
    return df["title"].tolist()

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
