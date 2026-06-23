# 📖 Semantic Book Recommender

A semantic book recommendation engine that finds books by **meaning**, not keyword matching. Describe the kind of story you want — *"a redemption story set against war"* — and it returns the most relevant books, filterable by category and emotional tone.

Built with sentence-transformer embeddings over a vector database, wrapped in a custom-styled Gradio interface.

## Demo

> _Add a screenshot or GIF of the running app here. (Take a screenshot of your browser at localhost:7860 and drag it into this README on GitHub.)_

## How it works

1. **Embeddings** — book descriptions are converted into vector representations using the `all-MiniLM-L6-v2` sentence-transformer model (runs locally, free, no API key needed).
2. **Vector search** — descriptions are stored in a Chroma vector database; a user query is embedded and matched against them via similarity search.
3. **Category filtering** — books are classified (fiction / non-fiction, etc.) so results can be narrowed.
4. **Emotional tone** — sentiment scores (joy, fear, anger, sadness, surprise) let users sort results by the *feeling* of a book.
5. **Interface** — a Gradio dashboard with a custom theme presents results as a cover gallery.

## Tech stack

Python · sentence-transformers (HuggingFace) · LangChain · ChromaDB · Gradio · pandas · numpy

## Running locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python GRADIO.py
```

Then open http://localhost:7860 in your browser.

**Note:** the dataset and generated files (`books_with_emotions.csv`, `tagged_description.txt`) are produced by the notebooks (`data-exploration.ipynb` and the classification/sentiment steps) and are not committed to keep the repo lightweight. Run the notebooks first to generate them, or use the "7k Books with Metadata" dataset from Kaggle as the starting point.

## What I learned

- How semantic / vector search differs from keyword search, and why embeddings capture meaning.
- Using a vector database (Chroma) for similarity retrieval.
- Applying zero-shot classification and sentiment analysis with pre-trained LLMs.
- Building and theming an interactive ML interface with Gradio.

## Possible improvements

- Wrap the recommender in a REST API (FastAPI) for reuse beyond the UI.
- Add evaluation metrics (precision@k) to measure recommendation quality.
- Deploy publicly (Hugging Face Spaces) and containerize with Docker.

## Evaluation

The recommender is evaluated offline using a hand-labeled query set
(`evaluate.py`) with standard information-retrieval metrics:

- **Precision@K** — fraction of the top-K recommendations that are relevant
- **Recall@K** — fraction of all relevant books found in the top-K

Run with: `python evaluate.py`

*Note: the relevance labels are author-assigned on a small query set, so
results are a coarse, internally-consistent benchmark rather than a
production-grade evaluation. A larger independently-labeled set would
give stronger numbers.*
