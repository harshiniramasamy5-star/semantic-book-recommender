import re
import pandas as pd
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

import gradio as gr


books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embeddings)


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [
        int(re.match(r"\d+", rec.page_content.strip('"')).group())
        for rec in recs
    ]
    book_recs = books[books["isbn13"].isin(books_list)].copy()

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]
    book_recs = book_recs.head(final_top_k)

    # Map dropdown tone -> emotion column in the CSV.
    # Only sort if the column actually exists, so a name mismatch can't crash.
    tone_col = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness",
    }.get(tone)

    if tone_col and tone_col in book_recs.columns:
        book_recs = book_recs.sort_values(by=tone_col, ascending=False)

    return book_recs


def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = str(row["description"]) if pd.notna(row["description"]) else ""
        truncated_description = " ".join(description.split()[:30]) + "..."

        authors_raw = str(row["authors"]) if pd.notna(row["authors"]) else "Unknown"
        authors_split = authors_raw.split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors_raw

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&family=Inter:wght@400;500;600&display=swap');

.gradio-container {
    background:
        radial-gradient(circle at 15% 0%, #f3e9d7 0%, transparent 45%),
        radial-gradient(circle at 90% 5%, #efe1cb 0%, transparent 40%),
        linear-gradient(160deg, #faf4e8 0%, #f1e6d2 60%, #ece0c9 100%) !important;
    font-family: 'Inter', sans-serif !important;
    min-height: 100vh;
    color: #3a2e22 !important;
}

/* ---------- Hero ---------- */
#hero { text-align: center; padding: 52px 20px 4px 20px; }
#hero h1 {
    font-family: 'Fraunces', serif !important;
    font-size: 3.1rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em;
    color: #4a2f1a !important;
    margin-bottom: 8px !important;
}
#hero h1 .accent { color: #9c4221; font-style: italic; }
#hero p {
    color: #6b5742 !important;
    font-size: 1.08rem !important;
    max-width: 720px; margin: 0 auto !important; line-height: 1.55;
    font-family: 'Fraunces', serif !important;
}
#badges { display:flex; justify-content:center; gap:8px; flex-wrap:wrap; margin:16px 0 4px; }
#badges span {
    font-size: 0.76rem; color:#7c4a23; letter-spacing:.02em;
    background: rgba(156,66,33,0.08);
    border:1px solid rgba(156,66,33,0.25);
    padding:4px 13px; border-radius:999px;
}

/* ---------- Search bar (control panel) ---------- */
#controls {
    background: #fffdf8 !important;
    border: 1px solid #e2d3ba !important;
    border-radius: 18px !important;
    padding: 18px !important;
    margin: 14px 14px 0 14px !important;
    box-shadow: 0 10px 30px rgba(90,60,30,0.12);
    position: relative; z-index: 50 !important; overflow: visible !important;
    align-items: flex-end !important;
}
.gradio-container .form { overflow: visible !important; background: transparent !important; }
#controls label span { color:#8a6a47 !important; font-weight:600 !important; }
#controls input, #controls .wrap {
    border-radius: 10px !important;
}

/* Gradio 6.x dropdown layering — z-index only, never position */
.gradio-container .wrap[data-testid="dropdown"],
.gradio-container [role="listbox"],
.gradio-container .options { z-index: 1000 !important; }
.gr-gallery { z-index: 1 !important; }

/* ---------- Search button ---------- */
#find-btn {
    background: linear-gradient(90deg, #b8541f, #9c4221) !important;
    border: none !important; color: #fff !important;
    font-weight: 700 !important; font-size: 1rem !important;
    border-radius: 12px !important; height: 56px !important;
    box-shadow: 0 6px 16px rgba(156,66,33,0.3);
    transition: all .15s ease;
}
#find-btn:hover { filter: brightness(1.08); transform: translateY(-2px); }

/* ---------- Shelf / gallery ---------- */
#section-title {
    font-family:'Fraunces',serif !important; color:#4a2f1a !important;
    margin: 22px 14px 2px 14px !important; font-size:1.3rem !important;
}
#gallery {
    margin: 4px 14px 30px 14px !important;
    background: #fffdf8 !important;
    border: 1px solid #e2d3ba !important;
    border-radius: 16px !important;
    padding: 14px !important;
    box-shadow: 0 10px 30px rgba(90,60,30,0.10);
    border-bottom: 6px solid #b8854f !important;
}
.gr-gallery { border-radius: 12px !important; }

footer { display:none !important; }
"""


with gr.Blocks(title="Semantic Book Recommender") as dashboard:

    with gr.Column(elem_id="hero"):
        gr.HTML("<h1>📖 Semantic Book <span class='accent'>Recommender</span></h1>")
        gr.Markdown(
            "Describe the kind of story you're craving — the engine finds it by *meaning*, "
            "not keywords, using sentence-transformer embeddings over a vector store."
        )
        gr.HTML(
            "<div id='badges'>"
            "<span>HuggingFace · all-MiniLM-L6-v2</span>"
            "<span>LangChain</span><span>ChromaDB</span><span>Gradio</span>"
            "</div>"
        )

    with gr.Row(elem_id="controls"):
        with gr.Column(scale=3):
            user_query = gr.Textbox(
                label="Describe a book",
                placeholder="e.g., A redemption story set against war",
                lines=1,
            )
        with gr.Column(scale=2):
            category_dropdown = gr.Dropdown(choices=categories, label="Category", value="All")
        with gr.Column(scale=2):
            tone_dropdown = gr.Dropdown(choices=tones, label="Emotional tone", value="All")
        with gr.Column(scale=1, min_width=150):
            submit_button = gr.Button("🔍 Search", elem_id="find-btn", variant="primary")

    gr.Markdown("### On the shelf", elem_id="section-title")
    output = gr.Gallery(
        elem_id="gallery",
        label="Recommended books",
        columns=5, rows=2, height="auto",
        object_fit="cover", show_label=False,
    )

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)
    user_query.submit(fn=recommend_books,
                      inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)


if __name__ == "__main__":
  dashboard.launch(theme=gr.themes.Soft(primary_hue="orange", secondary_hue="amber"), css=custom_css)
