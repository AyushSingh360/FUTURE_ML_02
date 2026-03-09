import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Handle relative import depending on where this script is called from
try:
    from text_preprocessing import preprocess_text
except ImportError:
    from src.text_preprocessing import preprocess_text

def load_system():
    try:
        # Resolve paths dynamically based on script location
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, 'models')
        
        vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
        cat_model = joblib.load(os.path.join(model_dir, 'category_model.pkl'))
        prio_model = joblib.load(os.path.join(model_dir, 'priority_model.pkl'))
        
        return vectorizer, cat_model, prio_model
    except Exception as e:
        print(f"Error loading models. Please ensure train_models.py has been run. Details: {e}")
        return None, None, None

def predict_ticket(text, vectorizer, cat_model, prio_model):
    """Takes a raw ticket string and returns the predicted category and priority."""
    # Preprocess
    clean_text = preprocess_text(text)
    
    # Vectorize
    X = vectorizer.transform([clean_text])
    
    # Predict
    category = cat_model.predict(X)[0]
    priority = prio_model.predict(X)[0]
    
    return category, priority

if __name__ == "__main__":
    vectorizer, cat_model, prio_model = load_system()
    if vectorizer:
        print("==================================================")
        print("    Support Ticket Classification System Ready    ")
        print("==================================================")
        
        # Example from requirements
        sample_input = "My payment failed but money was deducted"
        print(f"\nEvaluating Example Input:\n\"{sample_input}\"")
        
        cat, prio = predict_ticket(sample_input, vectorizer, cat_model, prio_model)
        print(f"\nResult:")
        print(f"Category: {cat}")
        print(f"Priority: {prio}")
        print("--------------------------------------------------")
        
        # Interactive mode
        print("\nEnter custom ticket text to predict (or type 'quit' to exit):")
        while True:
            try:
                user_input = input("\nTicket Text > ")
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                if user_input.strip() == "":
                    continue
                    
                cat, prio = predict_ticket(user_input, vectorizer, cat_model, prio_model)
                print(f"--> Predicted Category: {cat}")
                print(f"--> Predicted Priority: {prio}")
            except KeyboardInterrupt:
                break
                
        print("\nExiting system.")
