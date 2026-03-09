import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def plot_confusion_matrix(y_true, y_pred, classes, title, filename):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    print("Loading data and models for evaluation...")
    if not os.path.exists('../data/processed_tickets.csv'):
        print("Processed data not found. Please run train_models.py first.")
        return
        
    df = pd.read_csv('../data/processed_tickets.csv')
    
    vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')
    cat_model = joblib.load('../models/category_model.pkl')
    prio_model = joblib.load('../models/priority_model.pkl')
    
    X = vectorizer.transform(df['processed_text'].fillna(''))
    y_cat = df['category'].values
    y_prio = df['priority'].values
    
    # Use SAME random state to evaluate on the same test set
    X_train, X_test, y_train_c, y_test_c, y_train_p, y_test_p = train_test_split(
        X, y_cat, y_prio, test_size=0.2, random_state=42
    )
    
    os.makedirs('../visualizations', exist_ok=True)
    
    # Evaluate Category Model
    print("\n" + "="*50)
    print("--- Category Model Evaluation ---")
    print("="*50)
    cat_preds = cat_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test_c, cat_preds))
    print("\nClassification Report:\n", classification_report(y_test_c, cat_preds))
    
    cat_classes = np.unique(y_cat)
    plot_confusion_matrix(y_test_c, cat_preds, cat_classes, 
                          'Confusion Matrix - Category Prediction', 
                          '../visualizations/confusion_matrix.png')
    
    # Evaluate Priority Model
    print("\n" + "="*50)
    print("--- Priority Model Evaluation ---")
    print("="*50)
    prio_preds = prio_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test_p, prio_preds))
    print("\nClassification Report:\n", classification_report(y_test_p, prio_preds))
    
    prio_classes = np.unique(y_prio)
    plot_confusion_matrix(y_test_p, prio_preds, prio_classes, 
                          'Confusion Matrix - Priority Prediction', 
                          '../visualizations/priority_confusion_matrix.png')
                          
    print("\nEvaluation visualizations saved to ../visualizations/")

if __name__ == "__main__":
    main()
