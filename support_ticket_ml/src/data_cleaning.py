import pandas as pd

def load_data(filepath='../data/raw_tickets.csv'):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded dataset from {filepath} with shape {df.shape}")
        
        # Mapping Kaggle dataset columns to expected pipeline fields
        rename_map = {
            'Ticket Description': 'ticket_text',
            'Ticket Type': 'category',
            'Ticket Priority': 'priority',
            'Ticket ID': 'ticket_id'
        }
        df.rename(columns=rename_map, inplace=True)
        return df
    except FileNotFoundError:
        print(f"Error: Could not find file at {filepath}")
        return None

def clean_data(df):
    """Clean the data by handling missing values and duplicates."""
    initial_len = len(df)
    
    # Drop rows with missing text or target variables
    df = df.dropna(subset=['ticket_text', 'category', 'priority'])
    
    # Drop duplicates
    df = df.drop_duplicates(subset=['ticket_text'])
    
    final_len = len(df)
    if initial_len != final_len:
        print(f"Cleaned data: dropped {initial_len - final_len} rows (missing/duplicates). New shape: {df.shape}")
    else:
        print(f"Data is clean. No missing values or complete duplicates found. Shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        df_clean = clean_data(df)
        df_clean.to_csv('../data/processed_tickets.csv', index=False)
        print("Saved processed data to ../data/processed_tickets.csv")
