"""
app.py
------
Movie Recommendation System Using Reviews
College NLP Project • Review-2 Demonstration

Uses the supplied visual reference as inspiration with project-specific branding.
Includes:
- Dark Navy Sidebar Navigation
- Page 1: 🏠 Home with Hero Banner, Search, and 4 Feature Cards
- Page 2: 🎬 Browse Movies with Genre/Year/Sort Filters and 4-Column Poster Grid
- Page 3 & 4: ✨ Get Recommendations & Results with Sentiment Card, Keywords & Similar Movies
- Page 5: 💬 Sentiment Analysis with Donut Chart Distribution and Emotion Cards
- Page 6 & 7: ℹ️ About Project & Project Structure / Dataset Flow
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

# NLP and Recommendation modules
from preprocessing import clean_text, extract_top_keywords
from posters import find_movie_poster_urls, get_poster_url
from recommendation import (
    load_and_prepare_movies,
    build_recommender,
    get_all_genres,
    get_all_years,
    filter_and_browse_movies,
    get_movie_index,
    get_recommendations_by_title,
    recommend_by_review_or_text
)
from sentiment import (
    load_or_train_sentiment_model,
    predict_sentiment
)

# ---------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System Using Reviews",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS (Matches Reference Image Exactly)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Clean up default Streamlit padding & header */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1350px;
    }
    
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0b132b !important;
        color: #ffffff !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    @media (min-width: 768px) {
        [data-testid="stSidebar"][aria-expanded="false"] {
            display: block !important;
            flex: 0 0 16rem !important;
            width: 16rem !important;
            min-width: 16rem !important;
            margin-left: 0 !important;
            transform: translateX(0) !important;
            visibility: visible !important;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div {
            width: 16rem !important;
        }
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #94a3b8 !important;
    }
    
    /* Sidebar brand header */
    .sidebar-brand-box {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0 20px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 20px;
    }
    .brand-icon {
        width: 38px;
        height: 38px;
        background: linear-gradient(135deg, #4f46e5, #3b82f6);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
    }
    .brand-title {
        color: #ffffff !important;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .brand-subtitle {
        color: #64748b !important;
        font-size: 0.78rem;
        margin: 0;
    }
    
    /* Sidebar Navigation Radio Buttons */
    div[data-testid="stRadio"] > div {
        gap: 6px;
    }
    div[data-testid="stRadio"] label {
        background: transparent;
        padding: 10px 14px;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid transparent;
        display: flex;
        align-items: center;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #ffffff !important;
    }
    div[data-testid="stRadio"] label[data-checked="true"], 
    div[data-testid="stRadio"] label:has(input:checked) {
        background: #2563eb !important;
        color: #ffffff !important;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
    }
    div[data-testid="stRadio"] label[data-checked="true"] p,
    div[data-testid="stRadio"] label:has(input:checked) p {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Sidebar Footer */
    .sidebar-footer {
        padding-top: 30px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        font-size: 0.8rem;
        color: #64748b !important;
        margin-top: 40px;
    }
    
    /* Hero Section (Panel 1) */
    .hero-banner {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.82) 0%, rgba(15, 23, 0.95) 100%), 
                    radial-gradient(circle at 80% 20%, rgba(99, 102, 241, 0.25) 0%, transparent 50%),
                    #0b132b;
        border-radius: 18px;
        padding: 55px 40px 45px 40px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 15px 35px -5px rgba(11, 19, 43, 0.3);
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 12px;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        max-width: 650px;
        margin: 0 auto 28px auto;
        font-weight: 400;
        line-height: 1.5;
    }
    
    /* Feature Cards (Panel 1) */
    .feature-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 20px -3px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }
    .feature-icon-circle {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        margin-bottom: 16px;
    }
    .feature-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 8px;
    }
    .feature-desc {
        font-size: 0.85rem;
        color: #64748b;
        line-height: 1.4;
        margin: 0;
    }
    
    /* Movie Cards Grid (Panel 2 & 4) */
    .movie-grid-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    .movie-grid-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 18px -2px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }
    .poster-img {
        width: 100%;
        aspect-ratio: 2/3;
        object-fit: cover;
        border-radius: 10px;
        margin-bottom: 10px;
        background: #1e293b;
    }
    .card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 4px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .card-meta {
        font-size: 0.78rem;
        color: #64748b;
        margin: 0 0 6px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .card-bottom-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: auto;
        padding-top: 4px;
    }
    .card-rating {
        font-size: 0.82rem;
        font-weight: 700;
        color: #eab308;
    }
    .similar-badge {
        background: #4f46e5;
        color: #ffffff !important;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 6px;
    }
    
    /* Result Banners and Cards (Panel 4) */
    .success-alert {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        border-radius: 12px;
        padding: 14px 20px;
        font-weight: 600;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 22px;
    }
    
    .sentiment-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03);
    }
    .sentiment-result-inner {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-top: 10px;
    }
    .sentiment-circle-icon {
        width: 54px;
        height: 54px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
    }
    .pos-circle {
        background: #dcfce7;
        color: #15803d;
    }
    .neg-circle {
        background: #fee2e2;
        color: #b91c1c;
    }
    
    /* Keywords Badges */
    .keyword-pill {
        display: inline-block;
        background: #eff6ff;
        color: #2563eb;
        border: 1px solid #dbeafe;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    /* Action Buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5, #3b82f6) !important;
        border: none !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4338ca, #2563eb) !important;
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.4) !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Safe Dataset Loading & Caching (READ-ONLY from dataset/)
# ---------------------------------------------------------
MOVIES_CSV = os.path.join("dataset", "tmdb_5000_movies.csv")
CREDITS_CSV = os.path.join("dataset", "tmdb_5000_credits.csv")
IMDB_CSV = os.path.join("dataset", "IMDB Dataset.csv")


@st.cache_data(show_spinner="Loading TMDB 5000 movie dataset...")
def get_cached_movies(_poster_catalog_version: float):
    return load_and_prepare_movies(MOVIES_CSV, CREDITS_CSV)


@st.cache_resource(show_spinner="Computing TF-IDF vectorizer and Cosine Similarity matrix...")
def get_cached_recommender(movies_df):
    return build_recommender(movies_df, max_features=5000)


@st.cache_resource(show_spinner="Loading IMDb sentiment model (TF-IDF + Logistic Regression)...")
def get_cached_sentiment_model():
    return load_or_train_sentiment_model(IMDB_CSV)


@st.cache_data(ttl=60 * 60 * 24 * 30, show_spinner=False)
def get_cached_online_posters(movie_titles: tuple[tuple[str, str], ...]):
    return find_movie_poster_urls(movie_titles)


def resolve_movie_posters(movie_details):
    poster_urls = [
        get_poster_url(title, genre, year, rating)
        for title, year, genre, rating in movie_details
    ]
    uncatalogued_indices = [
        index for index, url in enumerate(poster_urls)
        if url.startswith("data:image/svg+xml")
    ]
    uncatalogued_titles = tuple(
        (movie_details[index][0], movie_details[index][1])
        for index in uncatalogued_indices
    )
    online_posters = get_cached_online_posters(uncatalogued_titles)
    for index, online_poster in zip(uncatalogued_indices, online_posters):
        if online_poster:
            poster_urls[index] = online_poster
    return poster_urls


def resolve_movie_poster(movie_details):
    return resolve_movie_posters([movie_details])[0]


# Verify dataset presence
for p in [MOVIES_CSV, CREDITS_CSV, IMDB_CSV]:
    if not os.path.exists(p):
        st.error(f"❌ Missing dataset file: `{p}`. Please verify your `dataset/` folder.")
        st.stop()

# Initialize data and models
movies_df = get_cached_movies(os.path.getmtime(os.path.join(os.path.dirname(__file__), "posters.py")))
tfidf_vectorizer, tfidf_matrix, similarity_matrix = get_cached_recommender(movies_df)
sentiment_clf, sentiment_vec, sentiment_metrics = get_cached_sentiment_model()

# Session State for cross-page navigation
if 'nav_page' not in st.session_state:
    st.session_state['nav_page'] = "🏠 Home"
if 'recommendation_input' not in st.session_state:
    st.session_state['recommendation_input'] = "The story was amazing and the acting was excellent."
if 'has_recommendations' not in st.session_state:
    st.session_state['has_recommendations'] = False
if 'home_search_submitted' not in st.session_state:
    st.session_state['home_search_submitted'] = False
if 'home_search_result_id' not in st.session_state:
    st.session_state['home_search_result_id'] = None


# ---------------------------------------------------------
# Sidebar Navigation (Styled to Match Reference Image)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-box">
        <div class="brand-icon">🎬</div>
        <div>
            <div class="brand-title">Movie Reviews</div>
            <div class="brand-subtitle">NLP Movie Discovery</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_options = [
        "🏠 Home",
        "🎬 Browse Movies",
        "✨ Get Recommendations",
        "💬 Sentiment Analysis",
        "ℹ️ About"
    ]
    
    # Sync navigation selection directly via session_state key
    selected_page = st.radio(
        label="Navigation Menu",
        options=nav_options,
        key="nav_page",
        label_visibility="collapsed"
    )

    st.markdown("""
    <div class="sidebar-footer">
        ❤️ Movies make life better 🎬
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# PAGE 1: 🏠 HOME PAGE (Panel 1)
# =========================================================
if st.session_state['nav_page'] == "🏠 Home":
    # Hero Section
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Movie Recommendation<br>System Using Reviews</div>
        <div class="hero-subtitle">
            Discover movies based on what people say, not just what they rate.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Search Bar in Home
    sc1, sc2, sc3 = st.columns([1, 6, 1])
    with sc2:
        search_col, btn_col = st.columns([5, 1])
        with search_col:
            home_search = st.text_input(
                "Search for a movie...",
                placeholder="Search for a movie (e.g., Inception, Avatar, Dark Knight)...",
                label_visibility="collapsed",
                key="home_movie_search"
            )
        with btn_col:
            if st.button("Search", type="primary", use_container_width=True):
                st.session_state['home_search_submitted'] = True
                matched_idx = get_movie_index(home_search, movies_df) if home_search.strip() else None
                st.session_state['home_search_result_id'] = (
                    int(movies_df.iloc[matched_idx]['id']) if matched_idx is not None else None
                )

    if st.session_state['home_search_submitted']:
        result_id = st.session_state['home_search_result_id']
        if result_id is None:
            if home_search.strip():
                st.warning(f"Movie not found: **{home_search.strip()}**. Check the title and try again.")
            else:
                st.info("Enter a movie title to search the TMDB catalog.")
        else:
            result = movies_df.loc[movies_df['id'] == result_id].iloc[0]
            result_poster = resolve_movie_poster((
                str(result['title']),
                str(result['year']),
                str(result['genres_display']),
                float(result['vote_average'])
            ))
            poster_col, details_col = st.columns([1, 3])
            with poster_col:
                st.markdown(
                    f'<img src="{result_poster}" alt="{result["title"]} poster" '
                    'style="width:100%;max-width:220px;border-radius:12px;">',
                    unsafe_allow_html=True
                )
            with details_col:
                st.subheader(str(result['title']))
                st.caption(
                    f"{result['year'] or 'Year unknown'} · {result['genres_display']} · "
                    f"Rating {float(result['vote_average']):.1f}/10"
                )
                st.write(result['overview'] or "No overview is available in the TMDB dataset.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 4 Feature Cards (Panel 1)
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-circle" style="background:#f3e8ff; color:#9333ea;">★</div>
            <div class="feature-title">Review Analysis</div>
            <p class="feature-desc">Understand opinions and audience impressions using NLP.</p>
        </div>
        """, unsafe_allow_html=True)
    with fc2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-circle" style="background:#dcfce7; color:#16a34a;">☺</div>
            <div class="feature-title">Sentiment Detection</div>
            <p class="feature-desc">Find positive & negative reviews with high-accuracy classification.</p>
        </div>
        """, unsafe_allow_html=True)
    with fc3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-circle" style="background:#dbeafe; color:#2563eb;">🎬</div>
            <div class="feature-title">Smart Recommendations</div>
            <p class="feature-desc">Get similar movies based on content, cast, and review patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    with fc4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-circle" style="background:#e0f2fe; color:#0284c7;">🗄</div>
            <div class="feature-title">TMDB + IMDb Data</div>
            <p class="feature-desc">Powered by 4,803 real movies and 50,000 authentic user reviews.</p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# PAGE 2: 🎬 BROWSE MOVIES (Panel 2)
# =========================================================
elif st.session_state['nav_page'] == "🎬 Browse Movies":
    st.markdown("<h2 style='color:#0f172a; margin-bottom:15px; font-weight:800;'>Browse Movies</h2>", unsafe_allow_html=True)

    # Top Filter Row
    f_search_col, f_genre_col, f_year_col, f_sort_col = st.columns([3, 2, 2, 2])

    with f_search_col:
        browse_search = st.text_input(
            "Search movie...",
            placeholder="🔍 Search movie...",
            label_visibility="collapsed"
        )

    all_genres = ["All"] + get_all_genres(movies_df)
    all_years = ["All"] + get_all_years(movies_df)

    with f_genre_col:
        browse_genre = st.selectbox("Genre", options=all_genres, index=0)
    with f_year_col:
        browse_year = st.selectbox("Year", options=all_years, index=0)
    with f_sort_col:
        browse_sort = st.selectbox("Sort by", options=["Popular", "Highest Rated", "Latest", "A-Z"], index=0)

    # Filter movies from TMDB dataset
    filtered_movies = filter_and_browse_movies(
        movies_df=movies_df,
        search_query=browse_search,
        selected_genre=browse_genre,
        selected_year=browse_year,
        sort_by=browse_sort,
        limit=16
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Render Movie Cards Grid (4 columns, matching Panel 2)
    if filtered_movies.empty:
        st.info("No movies matched your current filters. Try changing genre, year, or search query.")
    else:
        poster_details = [
            (
                str(movie['title']),
                str(movie['year']),
                str(movie['genres_display']),
                float(movie['vote_average'])
            )
            for _, movie in filtered_movies.iterrows()
        ]
        poster_by_index = dict(zip(
            filtered_movies.index,
            resolve_movie_posters(poster_details)
        ))

        # Display in rows of 4
        rows = [filtered_movies.iloc[i:i+4] for i in range(0, len(filtered_movies), 4)]
        for row in rows:
            cols = st.columns(4)
            for idx, (movie_index, movie) in enumerate(row.iterrows()):
                with cols[idx]:
                    poster = poster_by_index[movie_index]
                    title = movie['title']
                    year_val = movie['year']
                    genre_val = movie['genres_display']
                    rating_val = movie['vote_average']

                    st.markdown(f"""
                    <div class="movie-grid-card">
                        <img src="{poster}" class="poster-img" alt="{title}"/>
                        <div class="card-title" title="{title}">{title}</div>
                        <div class="card-meta">{year_val + ' • ' if year_val else ''}{genre_val}</div>
                        <div class="card-bottom-row">
                            <span class="card-rating">★ {rating_val:.1f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
# =========================================================
# PAGE 3 & 4: ✨ GET RECOMMENDATIONS & RESULTS (Panel 3 & 4)
# =========================================================
elif st.session_state['nav_page'] == "✨ Get Recommendations":
    st.markdown("<h2 style='color:#0f172a; margin-bottom:5px; font-weight:800;'>Get Recommendations</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:0.95rem; margin-bottom:20px;'>Enter a movie title or describe what you like to find movies with similar content.</p>", unsafe_allow_html=True)

    # Input Form (Panel 3)
    user_input = st.text_input(
        "Enter a movie you like (or a review):",
        value=st.session_state['recommendation_input'],
        placeholder="e.g. Inception OR The story was amazing and the acting was excellent."
    )

    get_rec_btn = st.button("Get Recommendations", type="primary")

    # Popular Movies Quick-Select Chips (Panel 3)
    st.markdown("<div style='margin-top:16px; margin-bottom:8px; font-size:0.9rem; font-weight:600; color:#475569;'>Or choose from popular movies:</div>", unsafe_allow_html=True)
    pop_cols = st.columns(5)
    popular_picks = ["Inception", "Interstellar", "The Dark Knight", "Forrest Gump", "Pulp Fiction"]
    for i, p_title in enumerate(popular_picks):
        with pop_cols[i]:
            if st.button(p_title, key=f"pop_chip_{i}", use_container_width=True):
                user_input = p_title
                st.session_state['recommendation_input'] = p_title
                st.session_state['has_recommendations'] = True
                st.rerun()

    # Trigger recommendation execution
    if get_rec_btn:
        st.session_state['recommendation_input'] = user_input
        st.session_state['has_recommendations'] = True

    # -----------------------------------------------------
    # PANEL 4: RECOMMENDATION RESULTS
    # -----------------------------------------------------
    if st.session_state['has_recommendations'] and st.session_state['recommendation_input'].strip():
        query_text = st.session_state['recommendation_input'].strip()
        
        # Prefer a matching TMDB title (exact or partial) over review analysis.
        matched_idx = get_movie_index(query_text, movies_df)
        is_movie_query = matched_idx is not None

        recs = []
        top_kws = []
        
        if is_movie_query:
            # Movie Title Recommendation
            matched_title = str(movies_df.iloc[matched_idx]['title'])
            selected_movie, recs = get_recommendations_by_title(
                matched_title, movies_df, similarity_matrix, top_n=5
            )
            top_kws = extract_top_keywords(selected_movie['overview'], n=5)
        else:
            # Free-text input is used for content recommendations only.
            recs, top_kws = recommend_by_review_or_text(query_text, movies_df, tfidf_vectorizer, tfidf_matrix, top_n=5)

        st.markdown("<br>", unsafe_allow_html=True)
        # Success Alert (Panel 4)
        st.markdown("""
        <div class="success-alert">
            <span>✔</span>
            <span>Based on your input, here are your recommended movies!</span>
        </div>
        """, unsafe_allow_html=True)

        if is_movie_query:
            st.info(f"Showing movies similar to **{selected_movie['title']}**.")
        kws_html = "".join([f"<span class='keyword-pill'>{kw}</span>" for kw in top_kws]) if top_kws else "<span style='color:#94a3b8; font-size:0.85rem;'>No dominant keywords detected.</span>"
        st.markdown(f"""
        <div class="sentiment-box">
            <div style="font-weight:700; font-size:1rem; color:#0f172a; margin-bottom:12px;">Top Keywords</div>
            <div style="margin-top:6px;">{kws_html}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#0f172a; font-weight:700;'>Recommended Movies</h3>", unsafe_allow_html=True)

        # 5 Recommended Movie Cards in a row (Panel 4)
        if recs:
            recommendation_poster_details = [
                (
                    str(movie['title']),
                    str(movie['year']),
                    str(movie['genres_display']),
                    float(movie['vote_average'])
                )
                for movie in recs
            ]
            recommendation_posters = resolve_movie_posters(recommendation_poster_details)
            rec_cols = st.columns(len(recs))
            for i, r in enumerate(recs):
                with rec_cols[i]:
                    r_poster = recommendation_posters[i]
                    r_title = r['title']
                    r_year = r['year']
                    r_genre = r['genres_display']
                    r_rating = r['vote_average']
                    r_match = r['match_percentage']

                    st.markdown(f"""
                    <div class="movie-grid-card">
                        <img src="{r_poster}" class="poster-img" alt="{r_title}"/>
                        <div class="card-title" title="{r_title}">{r_title}</div>
                        <div class="card-meta">{r_year + ' • ' if r_year else ''}{r_genre}</div>
                        <div class="card-bottom-row">
                            <span class="card-rating">★ {r_rating:.1f}</span>
                            <span class="similar-badge">Similar</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No recommendations found. Try another movie title or more descriptive keywords.")


# =========================================================
# PAGE 5: 💬 SENTIMENT ANALYSIS (Panel 5)
# =========================================================
elif st.session_state['nav_page'] == "💬 Sentiment Analysis":
    st.markdown("<h2 style='color:#0f172a; margin-bottom:5px; font-weight:800;'>Sentiment Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:0.95rem; margin-bottom:20px;'>Enter a movie review to check its sentiment.</p>", unsafe_allow_html=True)

    default_review = "The movie was boring and poorly written."
    review_input_val = st.text_area(
        label="Review Text",
        value=default_review,
        height=130,
        label_visibility="collapsed"
    )

    analyze_btn = st.button("Analyze Sentiment", type="primary")

    if analyze_btn or review_input_val:
        res = predict_sentiment(review_input_val, sentiment_clf, sentiment_vec)
        is_pos = res['sentiment'] == 'Positive'
        
        pos_pct = int(res['probability_positive'] * 100)
        neg_pct = int(res['probability_negative'] * 100)
        neutral_pct = 100 - (pos_pct + neg_pct)
        if neutral_pct < 0:
            neutral_pct = 0

        st.markdown("<br>", unsafe_allow_html=True)

        col_res1, col_res2 = st.columns(2)

        # Left Card: Sentiment Indicator (Panel 5)
        with col_res1:
            circle_cls = "pos-circle" if is_pos else "neg-circle"
            emoji_icon = "☺" if is_pos else "☹"
            sentiment_text = res['sentiment']
            desc_text = "The review expresses positive emotions and satisfaction." if is_pos else "The review expresses negative and dissatisfaction."

            st.markdown(f"""
            <div class="sentiment-box" style="display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; padding:35px 20px;">
                <div class="sentiment-circle-icon {circle_cls}" style="width:70px; height:70px; font-size:36px; margin-bottom:15px;">
                    {emoji_icon}
                </div>
                <div style="font-size:1.8rem; font-weight:800; color:{'#15803d' if is_pos else '#b91c1c'}; margin-bottom:6px;">
                    {sentiment_text}
                </div>
                <div style="font-size:0.9rem; color:#64748b; max-width:320px;">
                    {desc_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Right Card: Sentiment Distribution Donut Chart (Panel 5)
        with col_res2:
            # Generate pure SVG donut chart
            svg_donut = f"""
            <svg viewBox="0 0 160 160" width="130" height="130" style="transform: rotate(-90deg);">
                <!-- Background track -->
                <circle cx="80" cy="80" r="60" stroke="#f1f5f9" stroke-width="24" fill="transparent" />
                <!-- Positive green slice -->
                <circle cx="80" cy="80" r="60" stroke="#10b981" stroke-width="24" fill="transparent" 
                        stroke-dasharray="377" stroke-dashoffset="{377 - (377 * pos_pct / 100)}" />
                <!-- Negative red slice -->
                <circle cx="80" cy="80" r="60" stroke="#f43f5e" stroke-width="24" fill="transparent" 
                        stroke-dasharray="377" stroke-dashoffset="{377 - (377 * neg_pct / 100)}" 
                        stroke-dashoffset="0" style="transform-origin: center; transform: rotate({pos_pct * 3.6}deg);" />
                <!-- Center text -->
                <text x="80" y="86" font-family="-apple-system, sans-serif" font-size="20" font-weight="700" 
                      fill="#0f172a" text-anchor="middle" transform="rotate(90 80 80)">
                    {pos_pct if is_pos else neg_pct}%
                </text>
            </svg>
            """

            st.markdown(f"""
            <div class="sentiment-box" style="padding:25px;">
                <div style="font-weight:700; font-size:1rem; color:#0f172a; margin-bottom:20px;">Sentiment Distribution</div>
                <div style="display:flex; align-items:center; justify-content:space-around;">
                    <div>{svg_donut}</div>
                    <div style="font-size:0.9rem;">
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                            <span style="display:inline-block; width:12px; height:12px; background:#10b981; border-radius:3px;"></span>
                            <span style="color:#475569; font-weight:500;">Positive</span>
                            <span style="color:#0f172a; font-weight:700; margin-left:auto;">{pos_pct}%</span>
                        </div>
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                            <span style="display:inline-block; width:12px; height:12px; background:#f43f5e; border-radius:3px;"></span>
                            <span style="color:#475569; font-weight:500;">Negative</span>
                            <span style="color:#0f172a; font-weight:700; margin-left:auto;">{neg_pct}%</span>
                        </div>
                        <div style="display:flex; align-items:center; gap:8px;">
                            <span style="display:inline-block; width:12px; height:12px; background:#94a3b8; border-radius:3px;"></span>
                            <span style="color:#475569; font-weight:500;">Confidence</span>
                            <span style="color:#0f172a; font-weight:700; margin-left:auto;">{res['confidence_pct']}%</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# PAGE 6: ℹ️ ABOUT PROJECT (Panel 6 & 7)
# =========================================================
elif st.session_state['nav_page'] == "ℹ️ About":
    tab_about, tab_flow = st.tabs(["📘 About This Project", "🗂️ Project Structure & Dataset Flow"])

    # --- TAB 1: About This Project (Panel 6) ---
    with tab_about:
        st.markdown("<h2 style='color:#0f172a; margin-bottom:5px; font-weight:800;'>About This Project</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color:#475569; font-size:1rem; line-height:1.6; margin-bottom:25px;'>
            This Movie Recommendation System uses <strong>Natural Language Processing (NLP)</strong> to analyze movie reviews, 
            understand sentiment, and recommend movies based on similar patterns.
        </p>
        """, unsafe_allow_html=True)

        ab_c1, ab_c2 = st.columns(2)

        with ab_c1:
            st.markdown("""
            <div class="sentiment-box" style="padding:22px;">
                <h4 style="margin:0 0 15px 0; color:#0f172a; font-weight:700;">Key Technologies</h4>
                <div style="display:flex; flex-wrap:wrap; gap:10px;">
                    <span class="keyword-pill" style="font-size:0.85rem;">🐍 Python</span>
                    <span class="keyword-pill" style="font-size:0.85rem;">🐼 Pandas</span>
                    <span class="keyword-pill" style="font-size:0.85rem;">📚 NLTK</span>
                    <span class="keyword-pill" style="font-size:0.85rem;">⚙️ Scikit-learn</span>
                    <span class="keyword-pill" style="font-size:0.85rem;">👑 Streamlit</span>
                    <span class="keyword-pill" style="font-size:0.85rem;">🎬 TMDB + IMDb Datasets</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with ab_c2:
            st.markdown("""
            <div class="sentiment-box" style="padding:22px;">
                <h4 style="margin:0 0 15px 0; color:#0f172a; font-weight:700;">Dataset Details</h4>
                <div style="font-size:0.92rem; color:#334155; line-height:1.7;">
                    <div>📂 <strong>TMDB 5000:</strong> 4,803 movie records with overview, genres, cast, crew</div>
                    <div>💬 <strong>IMDb 50K:</strong> 50,000 labeled movie reviews (25k positive, 25k negative)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        ### 🎯 **Problem Statement**
        Traditional movie recommendation engines often rely solely on numerical star ratings or user IDs, ignoring the rich thematic content in movie plots, genre keywords, directing styles, and qualitative text reviews. This project bridges this gap using NLP.

        ### 🚀 **Project Objectives**
        - Perform text preprocessing (tokenization, stopword filtering, RegEx cleaning, and lemmatization) on movie metadata.
        - Implement **TF-IDF Vectorization** to represent movie semantics in a continuous vector space.
        - Compute pairwise **Cosine Similarity** across 4,803 TMDB movies.
        - Train a supervised **Logistic Regression** sentiment classifier on 50,000 IMDb movie reviews.
        - Deploy an intuitive web application following modern UI design principles.

        ### 🔮 **Future Enhancements**
        - Contextual BERT/Transformer embeddings (e.g. SBERT) for fine-grained phrase semantics.
        - Collaborative filtering integration for hybrid recommendations.
        - Real-time TMDB API trailer fetching.
        """)

    # --- TAB 2: Project Structure & Dataset Usage (Panel 7) ---
    with tab_flow:
        st.markdown("<h2 style='color:#0f172a; margin-bottom:15px; font-weight:800;'>Project Structure & Dataset Usage</h2>", unsafe_allow_html=True)

        p7_col1, p7_col2 = st.columns(2)

        with p7_col1:
            st.markdown("<h4 style='margin:0 0 12px 0; color:#0f172a; font-weight:700;'>Project Structure</h4>", unsafe_allow_html=True)
            st.code("""MovieRecommendationSystem/
|-- app.py                      (Streamlit web dashboard)
|-- recommendation.py           (Content similarity & browse logic)
|-- sentiment.py                (IMDb sentiment classification)
|-- preprocessing.py            (NLP text cleaning & tokenization)
|-- posters.py                  (Poster provider & SVG generator)
|
|-- dataset/                    [READ-ONLY ORIGINAL FILES]
|   |-- tmdb_5000_movies.csv    (4,803 movie metadata)
|   |-- tmdb_5000_credits.csv   (4,803 cast & crew records)
|   `-- IMDB Dataset.csv        (50,000 movie reviews)
|
|-- models/                     [SAVED MODEL ARTIFACTS]
|   |-- sentiment_model.pkl     (Trained Logistic Regression)
|   |-- sentiment_vectorizer.pkl (Fitted TF-IDF Vectorizer)
|   `-- sentiment_metrics.json  (Benchmark test evaluation)
|
|-- requirements.txt            (Project dependencies)
`-- README.md                   (Documentation & Review-2 guide)""", language=None)
        with p7_col2:
            st.markdown("""<div class="sentiment-box" style="padding:24px;">
<h4 style="margin:0 0 15px 0; color:#0f172a; font-weight:700;">How the Dataset is Used</h4>
                
<div style="background:#f0f9ff; border:1px solid #bae6fd; border-radius:10px; padding:14px; margin-bottom:10px;">
<div style="font-weight:700; color:#0369a1; font-size:0.95rem;">TMDB 5000</div>
<div style="font-size:0.85rem; color:#475569;">Movie title, genres, keywords, overview, poster link</div>
</div>
                
<div style="text-align:center; color:#64748b; font-size:1.2rem; margin:4px 0;">↓</div>
                
<div style="background:#faf5ff; border:1px solid #e9d5ff; border-radius:10px; padding:14px; margin-bottom:10px;">
<div style="font-weight:700; color:#7e22ce; font-size:0.95rem;">IMDb 50K</div>
<div style="font-size:0.85rem; color:#475569;">Review text, sentiment label (25k positive, 25k negative)</div>
</div>
                
<div style="text-align:center; color:#64748b; font-size:1.2rem; margin:4px 0;">↓</div>
                
<div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:14px;">
<div style="font-weight:700; color:#15803d; font-size:0.95rem;">Combined NLP Pipeline</div>
<div style="font-size:0.85rem; color:#475569;">Text preprocessing ➔ TF-IDF ➔ Sentiment Analysis ➔ Cosine Similarity ➔ Recommendations</div>
</div>
</div>""", unsafe_allow_html=True)
