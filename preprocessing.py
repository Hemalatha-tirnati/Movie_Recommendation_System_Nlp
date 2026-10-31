"""
preprocessing.py
----------------
Natural Language Processing (NLP) text preprocessing and data cleaning utilities
for the Movie Recommendation System and Sentiment Analysis.
Includes:
- Text normalization & cleaning (HTML stripping, RegEx, lowercasing)
- Tokenization
- Stopword removal
- Lemmatization (NLTK WordNetLemmatizer)
- Keyword extraction
- TMDB JSON metadata entity extraction
"""

import re
import ast
from typing import List, Any
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are available
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = set(stopwords.words('english'))

try:
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    LEMMATIZER = WordNetLemmatizer()


def clean_text(text: Any) -> str:
    if text is None or not isinstance(text, str):
        return ""
    cleaned = re.sub(r'<.*?>', ' ', text)
    # Remove URLs
    cleaned = re.sub(r'https?://\S+|www\.\S+', ' ', cleaned)
    # Convert to lowercase
    cleaned = cleaned.lower()
    # Keep only alphanumeric characters and spaces
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleaned)
    # Remove excessive whitespace
    cleaned = ' '.join(cleaned.split())
    return cleaned

def tokenize_text(text: str) -> List[str]:
    cleaned = clean_text(text)
    return cleaned.split()

def remove_stopwords(text: str) -> str:
    words = text.split()
    filtered_words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    return ' '.join(filtered_words)

def lemmatize_text(text: str) -> str:
    words = clean_text(text).split()
    try:
        return ' '.join(LEMMATIZER.lemmatize(word) for word in words)
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        try:
            return ' '.join(LEMMATIZER.lemmatize(word) for word in words)
        except LookupError:
            return ' '.join(words)

def extract_top_keywords(text: str, n: int = 5) -> List[str]:
    """
    Extracts top informative keywords from a user query or review.
    Filters out stopwords and short tokens, ordering by frequency and length.
    """
    cleaned = clean_text(text)
    words = cleaned.split()
    meaningful = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    
    # Frequency count while preserving distinct order
    counts = {}
    for w in meaningful:
        counts[w] = counts.get(w, 0) + 1
    
    # Sort by frequency, then word length
    sorted_words = sorted(counts.keys(), key=lambda w: (counts[w], len(w)), reverse=True)
    return sorted_words[:n]


def parse_json_column(column_val: Any, key: str = 'name', limit: int = None) -> List[str]:
    if column_val is None or not isinstance(column_val, str) or column_val.strip() == "":
        return []
    try:
        data = ast.literal_eval(column_val)
        if not isinstance(data, list):
            return []
        extracted = [item[key] for item in data if isinstance(item, dict) and key in item]
        return extracted[:limit] if limit is not None else extracted
    except (ValueError, SyntaxError, TypeError):
        return []


def extract_director(crew_val: Any) -> List[str]:
    if crew_val is None or not isinstance(crew_val, str) or crew_val.strip() == "":
        return []
    try:
        data = ast.literal_eval(crew_val)
        if not isinstance(data, list):
            return []
        for item in data:
            if isinstance(item, dict) and item.get('job') == 'Director' and 'name' in item:
                return [item['name']]
        return []
    except (ValueError, SyntaxError, TypeError):
        return []


def create_combined_tags(overview: str, genres: List[str], keywords: List[str], 
                         cast: List[str], director: List[str]) -> str:
    clean_overview = clean_text(overview)
    processed_genres = [re.sub(r'\s+', '', g.lower()) for g in genres if g]
    processed_keywords = [re.sub(r'\s+', '', k.lower()) for k in keywords if k]
    processed_cast = [re.sub(r'\s+', '', c.lower()) for c in cast if c]
    processed_director = [re.sub(r'\s+', '', d.lower()) for d in director if d]
    combined = (
        clean_overview + " " +
        " ".join(processed_genres) + " " +
        " ".join(processed_keywords) + " " +
        " ".join(processed_cast) + " " +
        " ".join(processed_director)
    )
    return remove_stopwords(clean_text(combined))
