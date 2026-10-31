# 🎬 Movie Recommendation System Using Reviews

A complete Natural Language Processing (NLP) and Machine Learning project designed and implemented for College Review-2.

This project delivers:
1. **Content-Based Movie Recommendation Engine** powered by TMDB 5000 Movies & Credits datasets using **TF-IDF Vectorization** and **Cosine Similarity**.
2. **Review Sentiment Analysis System** trained on **50,000 real IMDb Movie Reviews** using **TF-IDF (Unigrams + Bigrams)** and **Logistic Regression** (~89.7% test accuracy).
3. **Review/Vibe-based Recommendation** where users can input natural language plot preferences or reviews to discover matching films.
4. **Browse Movies Catalog** with live filters for Genre, Release Year, and Sorting.
5. **Interactive Streamlit Web Dashboard** inspired by the supplied visual reference, with project-specific branding, a dark-navy sidebar, responsive poster cards, and sentiment visualizations.

---

## 📁 Project Structure

```
MovieRecommendationSystem/
│
├── app.py                      # Multi-page Streamlit web dashboard
├── recommendation.py           # Content similarity, browse filters, and search logic
├── sentiment.py                # IMDb sentiment classification model & inference
├── preprocessing.py            # NLP cleaning, stopwords, tokenization & lemmatization
├── posters.py                  # Poster provider & SVG poster fallback generator
│
├── dataset/                    # READ-ONLY ORIGINAL DATASETS
│   ├── tmdb_5000_movies.csv    # 4,803 TMDB movies with metadata
│   ├── tmdb_5000_credits.csv   # 4,803 cast & crew records
│   └── IMDB Dataset.csv        # 50,000 labeled movie reviews
│
├── models/                     # SAVED SERIALIZED MODEL ARTIFACTS
│   ├── sentiment_model.pkl     # Pre-trained Logistic Regression model
│   ├── sentiment_vectorizer.pkl# Fitted TF-IDF vectorizer (10k features)
│   └── sentiment_metrics.json  # Model evaluation benchmark metrics
│
├── requirements.txt            # Python project dependencies
└── README.md                   # Project documentation & presentation guide
```

---

## 🎨 UI Reference Alignment (7 Panels)

The application faithfully implements all seven visual panels from the design reference:

1. **Panel 1 — Home Page:**
   - Dark gradient hero banner with title, subtitle, and instant movie search.
   - 4 Feature Cards: *Review Analysis*, *Sentiment Detection*, *Smart Recommendations*, *TMDB + IMDb Data*.
2. **Panel 2 — Browse Movies:**
   - Search bar with Genre, Year, and Sort by (`Popular`, `Highest Rated`, `Latest`, `A-Z`) dropdowns.
   - 4-column responsive movie card grid with posters, titles, release years, genres, and ratings.
3. **Panel 3 — Get Recommendations:**
   - Input field accepting movie titles or free-form reviews/preferences.
   - Quick-select pill buttons for popular movies (*Inception*, *Interstellar*, *The Dark Knight*, *Forrest Gump*, *Pulp Fiction*).
4. **Panel 4 — Recommendation Results:**
   - Success alert banner: `✔ Based on your input, here are your recommended movies!`.
   - Side-by-side cards: **Your Sentiment** (Positive/Negative badge) and **Top Keywords** (extracted NLP terms).
   - Horizontal row of recommended movie cards with match badges.
5. **Panel 5 — Sentiment Analysis:**
   - Dedicated review text area with preset examples.
   - Side-by-side result cards: **Emotion Indicator** (smile/sad face) and **Sentiment Distribution** (SVG donut chart).
6. **Panel 6 — About Page:**
   - Problem statement, project objectives, key technologies, and dataset summaries.
7. **Panel 7 — Project Structure & Dataset Usage:**
   - Complete project architecture tree and visual workflow diagram tracing data flow from raw CSVs to recommendations.

---

## 🧠 NLP Techniques Implemented

| NLP Technique | Purpose | Implementation Details |
| :--- | :--- | :--- |
| **Text Cleaning & Normalization** | Remove noise from raw text | Lowercasing, HTML stripping (`<br />`), URL removal, RegEx punctuation stripping |
| **Stopword Removal** | Eliminate non-informative tokens | NLTK English stopwords list (198 tokens) |
| **Lemmatization** | Reduce words to root dictionary form | NLTK `WordNetLemmatizer` |
| **Keyword Extraction** | Identify salient topics in reviews | Frequency and length-based term ranking filtered by non-stopwords |
| **Metadata Entity Extraction** | Structure JSON fields | Safely parsing `genres`, `keywords`, `cast` (top 4), and `crew` (Director) |
| **Feature Combination** | Unify movie attributes | Concatenating overview, genres, cast, and director into single semantic tags |
| **TF-IDF Vectorization** | Text to numerical representation | $TF(t, d) \times IDF(t, D)$ with unigram & bigram support (`ngram_range=(1,2)`) |
| **Cosine Similarity** | Angular proximity calculation | $\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$ across all 4,803 movie vectors |
| **Supervised Sentiment Classification** | Binary opinion classification | Logistic Regression trained on 50k IMDb reviews with probability calibration |

---

## 🔄 System Architecture Workflow

```
   [tmdb_5000_movies.csv] + [tmdb_5000_credits.csv]            [IMDB Dataset.csv] (50k Reviews)
                         │                                                   │
                         ▼                                                   ▼
            Merge Datasets on Movie ID                              Text Cleaning & Regex
                         │                                                   │
                         ▼                                                   ▼
             Parse JSON Entities & Tags                             NLTK Stopword Removal
       (Overview + Genres + Cast + Director)                                │
                         │                                                   ▼
                         ▼                                     TF-IDF Vectorizer (10k N-grams)
           TF-IDF Vectorizer (5k Features)                                   │
                         │                                                   ▼
                         ▼                                      Logistic Regression Classifier
             Cosine Similarity Matrix                                        │
                         │                                                   ▼
                         ▼                                      Sentiment: Positive / Negative
             Top-N Movie Recommendations                             (Confidence & Probabilities)
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10+ installed
- Terminal / PowerShell

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 🎓 College Review-2 Presentation Notes

- **Dataset Integrity:** All three files in `dataset/` remain 100% read-only.
- **Fast Performance:** Model training on 50k IMDb reviews is cached persistently in `models/` (<0.1s startup time).
- **Posters:** Uses curated poster URLs first, then looks up missing movie thumbnails from Wikipedia in one cached batch for both Browse Movies and recommendations. Generated SVG artwork remains as a fallback when no thumbnail is available.
- **Dual Recommender:** Supports both exact movie title matching and review/vibe natural language search.
