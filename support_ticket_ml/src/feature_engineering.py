import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def create_features(df, text_column='processed_text', max_features=5000):
    """Convert text data into TF-IDF features."""
    print("Extracting features using TF-IDF...")
    
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df[text_column])
    
    # Save the vectorizer for later use in prediction
    os.makedirs('../models', exist_ok=True)
    joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')
    print("Saved TF-IDF vectorizer to ../models/tfidf_vectorizer.pkl")
    
    return X, vectorizer

def encode_labels(df, category_col='category', priority_col='priority'):
    """Extract labels for classification."""
    y_category = df[category_col].values
    y_priority = df[priority_col].values
    
    return y_category, y_priority
