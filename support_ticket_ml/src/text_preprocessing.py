import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    """Clean and preprocess a single text string."""
    if not isinstance(text, str):
        return ""
        
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenization and Stop words removal
    try:
        stop_words = set(stopwords.words('english'))
    except OSError:
        # Fallback if download failed
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Rejoin tokens
    return ' '.join(tokens)

def process_dataframe(df, text_column='ticket_text'):
    """Apply preprocessing to a dataframe column."""
    df_processed = df.copy()
    print("Preprocessing text...")
    df_processed['processed_text'] = df_processed[text_column].apply(preprocess_text)
    print("Text preprocessing completed.")
    return df_processed
