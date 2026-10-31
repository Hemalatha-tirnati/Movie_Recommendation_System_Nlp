"""
sentiment.py
------------
NLP Sentiment Analysis model trained on the IMDb 50,000 Movie Reviews dataset.
Uses TF-IDF Vectorization and Logistic Regression for binary sentiment classification (Positive / Negative).
Includes persistent caching to avoid retraining on Streamlit reruns.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from preprocessing import clean_text


MODELS_DIR = "models"
MODEL_FILE = os.path.join(MODELS_DIR, "sentiment_model.pkl")
VECTORIZER_FILE = os.path.join(MODELS_DIR, "sentiment_vectorizer.pkl")
METRICS_FILE = os.path.join(MODELS_DIR, "sentiment_metrics.json")


def train_sentiment_model(
    imdb_csv_path: str,
    max_features: int = 10000,
    models_dir: str = MODELS_DIR
) -> Tuple[LogisticRegression, TfidfVectorizer, Dict[str, Any]]:

    if not os.path.exists(imdb_csv_path):
        raise FileNotFoundError(f"IMDb dataset not found at: {imdb_csv_path}")
    os.makedirs(models_dir, exist_ok=True)
    df = pd.read_csv(imdb_csv_path)
    
    # 2. Preprocess review text
    df['clean_review'] = df['review'].apply(clean_text)

    # 3. Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_review'],
        df['sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment']
    )

    # 4. TF-IDF Feature Extraction with unigrams and bigrams
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5. Train Logistic Regression Classifier
    classifier = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    classifier.fit(X_train_vec, y_train)

    # 6. Evaluate Model Performance
    y_pred = classifier.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label='positive')
    cm = confusion_matrix(y_test, y_pred, labels=['negative', 'positive'])

    metrics = {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'accuracy': round(float(acc), 4),
        'accuracy_pct': round(float(acc) * 100, 2),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1_score': round(float(f1), 4),
        'confusion_matrix': cm.tolist(),
        'vocabulary_size': len(vectorizer.vocabulary_)
    }

    # 7. Persist Artifacts to Disk
    model_path = os.path.join(models_dir, "sentiment_model.pkl")
    vec_path = os.path.join(models_dir, "sentiment_vectorizer.pkl")
    metric_path = os.path.join(models_dir, "sentiment_metrics.json")

    joblib.dump(classifier, model_path)
    joblib.dump(vectorizer, vec_path)
    with open(metric_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    return classifier, vectorizer, metrics


def load_or_train_sentiment_model(
    imdb_csv_path: str = "dataset/IMDB Dataset.csv",
    models_dir: str = MODELS_DIR
) -> Tuple[LogisticRegression, TfidfVectorizer, Dict[str, Any]]:
    """
    Loads pre-trained sentiment classifier if available; otherwise trains and persists it.
    """
    model_path = os.path.join(models_dir, "sentiment_model.pkl")
    vec_path = os.path.join(models_dir, "sentiment_vectorizer.pkl")
    metric_path = os.path.join(models_dir, "sentiment_metrics.json")

    if os.path.exists(model_path) and os.path.exists(vec_path) and os.path.exists(metric_path):
        try:
            classifier = joblib.load(model_path)
            vectorizer = joblib.load(vec_path)
            with open(metric_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            return classifier, vectorizer, metrics
        except Exception:
            # If loading failed for any reason, re-train
            pass

    # If artifacts do not exist, train from dataset
    return train_sentiment_model(imdb_csv_path, models_dir=models_dir)


def predict_sentiment(
    review_text: str,
    classifier: LogisticRegression,
    vectorizer: TfidfVectorizer
) -> Dict[str, Any]:
    """
    Predicts the sentiment of a given user review.
    Returns:
    - sentiment: "Positive" or "Negative"
    - confidence: float (0.0 to 1.0)
    - probability_positive: float
    - probability_negative: float
    - key_influential_words: list of words from review that influenced the classification
    """
    cleaned = clean_text(review_text)
    if not cleaned:
        return {
            'sentiment': 'Neutral / Empty',
            'confidence': 0.0,
            'probability_positive': 0.5,
            'probability_negative': 0.5,
            'key_words': []
        }

    # Transform input review
    vec = vectorizer.transform([cleaned])
    pred = classifier.predict(vec)[0]
    probabilities = classifier.predict_proba(vec)[0]
    
    # Class mapping
    classes = list(classifier.classes_)
    pos_idx = classes.index('positive') if 'positive' in classes else 1
    neg_idx = classes.index('negative') if 'negative' in classes else 0

    prob_pos = float(probabilities[pos_idx])
    prob_neg = float(probabilities[neg_idx])
    
    sentiment_label = "Positive" if pred == 'positive' else "Negative"
    confidence = prob_pos if pred == 'positive' else prob_neg

    # Extract influential terms
    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]
    non_zero_indices = vec.nonzero()[1]

    word_influences = []
    for idx in non_zero_indices:
        word = feature_names[idx]
        weight = coefs[idx]
        word_influences.append((word, weight))

    # Sort words by weight matching the predicted sentiment
    if pred == 'positive':
        word_influences = sorted(word_influences, key=lambda x: x[1], reverse=True)
    else:
        word_influences = sorted(word_influences, key=lambda x: x[1], reverse=False)

    top_words = [w[0] for w in word_influences[:6]]

    return {
        'sentiment': sentiment_label,
        'confidence': round(confidence, 4),
        'confidence_pct': round(confidence * 100, 2),
        'probability_positive': round(prob_pos, 4),
        'probability_negative': round(prob_neg, 4),
        'probability_positive_pct': round(prob_pos * 100, 1),
        'probability_negative_pct': round(prob_neg * 100, 1),
        'key_words': top_words
    }


if __name__ == "__main__":
    imdb_csv = os.path.join("dataset", "IMDB Dataset.csv")
    print("Loading or training IMDb Sentiment Model...")
    clf, vec, metrics = load_or_train_sentiment_model(imdb_csv)
    print(f"Model Ready! Accuracy: {metrics['accuracy_pct']}% | Samples: {metrics['total_samples']}")
    
    sample = "The story was amazing and the acting was excellent."
    print(f"\nAnalyzing sample review: '{sample}'")
    pred = predict_sentiment(sample, clf, vec)
    print(f"Result: {pred['sentiment']} (Confidence: {pred['confidence_pct']}%)")
    print(f"Top influential terms: {pred['key_words']}")


