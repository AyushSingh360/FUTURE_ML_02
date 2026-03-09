import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from data_cleaning import load_data, clean_data
from text_preprocessing import process_dataframe
from feature_engineering import create_features, encode_labels

def train_and_evaluate(X_train, y_train, X_test, y_test, models, target_name):
    print(f"\n--- Training Models for {target_name} Prediction ---")
    best_model = None
    best_acc = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    print(f"-> Best model for {target_name}: {best_name} with Accuracy {best_acc:.4f}")
    return best_model, best_name

def main():
    print("1. Loading Data...")
    df = load_data('../data/raw_tickets.csv')
    if df is None:
        print("Data not found. Please run data_generation.py first.")
        return
        
    df = clean_data(df)
    
    print("\n2. Text Preprocessing...")
    df = process_dataframe(df)
    # Save processed dataframe for EDA later
    df.to_csv('../data/processed_tickets.csv', index=False)
    
    print("\n3. Feature Extraction...")
    X, vectorizer = create_features(df, max_features=5000)
    y_category, y_priority = encode_labels(df)
    
    print("\n4. Train/Test Split (80/20)...")
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_category, test_size=0.2, random_state=42)
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_priority, test_size=0.2, random_state=42)
    
    # 5. Define Models
    cat_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Multinomial NB': MultinomialNB()
    }
    
    priority_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # 6. Train Models
    best_cat_model, _ = train_and_evaluate(X_train_c, y_train_c, X_test_c, y_test_c, cat_models, "Category")
    best_prio_model, _ = train_and_evaluate(X_train_p, y_train_p, X_test_p, y_test_p, priority_models, "Priority")
    
    # 7. Save Best Models
    os.makedirs('../models', exist_ok=True)
    joblib.dump(best_cat_model, '../models/category_model.pkl')
    joblib.dump(best_prio_model, '../models/priority_model.pkl')
    print("\nModels successfully saved to ../models/")

if __name__ == "__main__":
    main()
