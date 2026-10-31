"""
recommendation.py
-----------------
Content-Based Movie Recommendation System using TMDB 5000 Movies & Credits datasets.
Implements TF-IDF vectorization and Cosine Similarity calculation.
Also provides browse filtering (by genre, year, sorting) and review-based recommendation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple, Optional

from preprocessing import (
    parse_json_column,
    extract_director,
    create_combined_tags,
    clean_text,
    remove_stopwords,
    extract_top_keywords
)
from posters import get_poster_url


def load_and_prepare_movies(movies_path: str, credits_path: str) -> pd.DataFrame:
    """
    Loads TMDB movies and credits datasets, merges them on movie ID,
    extracts structured entities, and constructs the NLP combined tags.
    """
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"Movies dataset not found at: {movies_path}")
    if not os.path.exists(credits_path):
        raise FileNotFoundError(f"Credits dataset not found at: {credits_path}")

    # Read CSVs (READ-ONLY from dataset/ as required)
    movies_df = pd.read_csv(movies_path)
    credits_df = pd.read_csv(credits_path)

    # Merge on movie id to preserve all 4,803 movies
    df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', suffixes=('', '_credit'))

    # Fill missing values
    df['overview'] = df['overview'].fillna('')
    df['release_date'] = df['release_date'].fillna('Unknown')
    df['vote_average'] = df['vote_average'].fillna(0.0)
    df['vote_count'] = df['vote_count'].fillna(0)
    df['popularity'] = df['popularity'].fillna(0.0)

    # Extract Year from release_date
    df['year'] = df['release_date'].apply(
        lambda d: str(d).split('-')[0] if str(d) != 'Unknown' and '-' in str(d) else ''
    )

    # Parse JSON entities
    df['parsed_genres'] = df['genres'].apply(lambda x: parse_json_column(x, key='name'))
    df['primary_genre'] = df['parsed_genres'].apply(lambda g: g[0] if g else 'General')
    df['genres_display'] = df['parsed_genres'].apply(lambda g: ', '.join(g[:3]) if g else 'General')

    df['parsed_keywords'] = df['keywords'].apply(lambda x: parse_json_column(x, key='name', limit=10))
    df['parsed_cast'] = df['cast'].apply(lambda x: parse_json_column(x, key='name', limit=4))
    df['parsed_director'] = df['crew'].apply(extract_director)

    # Create combined NLP tags
    df['combined_tags'] = df.apply(
        lambda row: create_combined_tags(
            overview=row['overview'],
            genres=row['parsed_genres'],
            keywords=row['parsed_keywords'],
            cast=row['parsed_cast'],
            director=row['parsed_director']
        ),
        axis=1
    )

    # Attach poster URLs
    df['poster_url'] = df.apply(
        lambda r: get_poster_url(
            title=r['title'],
            genre=r['primary_genre'],
            year=r['year'],
            rating=float(r['vote_average'])
        ),
        axis=1
    )

    clean_cols = [
        'id', 'title', 'year', 'overview', 'release_date', 'vote_average', 'vote_count',
        'popularity', 'parsed_genres', 'primary_genre', 'genres_display',
        'parsed_keywords', 'parsed_cast', 'parsed_director', 'poster_url', 'combined_tags'
    ]
    result_df = df[clean_cols].copy()
    result_df.reset_index(drop=True, inplace=True)
    return result_df


def build_recommender(movies_df: pd.DataFrame, max_features: 
                      int = 5000) -> Tuple[TfidfVectorizer, Any, np.ndarray]:
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['combined_tags'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, similarity_matrix


def get_all_genres(movies_df: pd.DataFrame) -> List[str]:
    """
    Returns unique list of all movie genres present in the dataset.
    """
    genres = set()
    for glist in movies_df['parsed_genres']:
        genres.update(glist)
    return sorted(list(genres))


def get_all_years(movies_df: pd.DataFrame) -> List[str]:
    """
    Returns unique list of release years sorted descending.
    """
    years = [y for y in movies_df['year'].unique() if y and y.isdigit()]
    return sorted(years, reverse=True)


def filter_and_browse_movies(
    movies_df: pd.DataFrame,
    search_query: str = "",
    selected_genre: str = "All",
    selected_year: str = "All",
    sort_by: str = "Popular",
    limit: int = 16
) -> pd.DataFrame:
    """
    Filters and sorts movies for the Browse Movies page.
    """
    filtered = movies_df.copy()

    # Search filter
    if search_query.strip():
        q = search_query.strip().lower()
        filtered = filtered[filtered['title'].str.lower().str.contains(q, regex=False)]

    # Genre filter
    if selected_genre != "All":
        filtered = filtered[filtered['parsed_genres'].apply(lambda glist: selected_genre in glist)]

    # Year filter
    if selected_year != "All":
        filtered = filtered[filtered['year'] == selected_year]

    # Sorting
    if sort_by == "Popular":
        filtered = filtered.sort_values(by='popularity', ascending=False)
    elif sort_by == "Highest Rated":
        # Filter out movies with very few votes for meaningful top rated lists
        filtered = filtered[filtered['vote_count'] >= 50].sort_values(by='vote_average', ascending=False)
    elif sort_by == "Latest":
        filtered = filtered.sort_values(by='release_date', ascending=False)
    elif sort_by == "A-Z":
        filtered = filtered.sort_values(by='title', ascending=True)

    return filtered.head(limit)


def get_movie_index(query_title: str, movies_df: pd.DataFrame) -> Optional[int]:
    """
    Finds index of movie by title.
    """
    query_clean = query_title.strip().lower()
    matches = movies_df[movies_df['title'].str.strip().str.lower() == query_clean]
    if not matches.empty:
        return matches.index[0]
    
    partial = movies_df[movies_df['title'].str.lower().str.contains(query_clean, regex=False)]
    if not partial.empty:
        return partial.index[0]
    return None


def get_recommendations_by_title(
    movie_title: str,
    movies_df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    top_n: int = 5
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    idx = get_movie_index(movie_title, movies_df)
    if idx is None:
        return None, []

    selected_row = movies_df.iloc[idx]
    selected_movie = {
        'id': int(selected_row['id']),
        'title': str(selected_row['title']),
        'year': str(selected_row['year']),
        'overview': str(selected_row['overview']),
        'genres': list(selected_row['parsed_genres']),
        'genres_display': str(selected_row['genres_display']),
        'cast': list(selected_row['parsed_cast']),
        'director': list(selected_row['parsed_director']),
        'release_date': str(selected_row['release_date']),
        'vote_average': float(selected_row['vote_average']),
        'vote_count': int(selected_row['vote_count']),
        'poster_url': str(selected_row['poster_url'])
    }

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for item_idx, score in sim_scores:
        if item_idx == idx:
            continue
        
        row = movies_df.iloc[item_idx]
        recommendations.append({
            'id': int(row['id']),
            'title': str(row['title']),
            'year': str(row['year']),
            'overview': str(row['overview']),
            'genres': list(row['parsed_genres']),
            'genres_display': str(row['genres_display']),
            'director': list(row['parsed_director']),
            'cast': list(row['parsed_cast']),
            'release_date': str(row['release_date']),
            'vote_average': float(row['vote_average']),
            'poster_url': str(row['poster_url']),
            'similarity_score': round(float(score), 4),
            'match_percentage': round(float(score) * 100, 1)
        })

        if len(recommendations) >= top_n:
            break

    return selected_movie, recommendations


def recommend_by_review_or_text(
    review_query: str,
    movies_df: pd.DataFrame,
    tfidf_vectorizer: TfidfVectorizer,
    tfidf_matrix: Any,
    top_n: int = 5
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Recommends movies based on natural language review or user preference description.
    Also extracts top keywords from the review query.
    """
    keywords = extract_top_keywords(review_query, n=6)
    cleaned_query = remove_stopwords(clean_text(review_query))
    if not cleaned_query.strip():
        return [], keywords

    query_vector = tfidf_vectorizer.transform([cleaned_query])
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    top_indices = cosine_sim.argsort()[::-1][:top_n]
    
    recommendations = []
    for idx in top_indices:
        score = cosine_sim[idx]
        row = movies_df.iloc[idx]
        recommendations.append({
            'id': int(row['id']),
            'title': str(row['title']),
            'year': str(row['year']),
            'overview': str(row['overview']),
            'genres': list(row['parsed_genres']),
            'genres_display': str(row['genres_display']),
            'director': list(row['parsed_director']),
            'cast': list(row['parsed_cast']),
            'release_date': str(row['release_date']),
            'vote_average': float(row['vote_average']),
            'poster_url': str(row['poster_url']),
            'similarity_score': round(float(score), 4),
            'match_percentage': round(float(score) * 100, 1)
        })

    return recommendations, keywords


if __name__ == "__main__":
    movies_csv = os.path.join("dataset", "tmdb_5000_movies.csv")
    credits_csv = os.path.join("dataset", "tmdb_5000_credits.csv")
    print("Loading and preparing TMDB movie datasets...")
    df = load_and_prepare_movies(movies_csv, credits_csv)
    print(f"Successfully loaded and preprocessed {len(df)} movies.")
    
    print("Building TF-IDF model and Cosine Similarity matrix...")
    tfidf_vec, tfidf_mat, sim_mat = build_recommender(df)
    print(f"TF-IDF matrix shape: {tfidf_mat.shape}")

    test_movie = "The Dark Knight"
    print(f"\nTesting recommendation for: '{test_movie}'")
    selected, recs = get_recommendations_by_title(test_movie, df, sim_mat, top_n=5)
    if selected:
        print(f"Selected Movie: {selected['title']} ({selected['year']}) [Rating: {selected['vote_average']}/10]")
        print("Top 5 Recommendations:")
        for r in recs:
            print(f"  - {r['title']} ({r['year']}) | Similarity: {r['similarity_score']} ({r['match_percentage']}%)")

