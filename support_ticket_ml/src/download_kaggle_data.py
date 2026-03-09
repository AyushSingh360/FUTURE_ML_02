import kagglehub
import shutil
import os

print("Downloading dataset...")
path = kagglehub.dataset_download("suraj520/customer-support-ticket-dataset")
print("Path to dataset files:", path)

files = os.listdir(path)
print("Files in dataset:", files)

# Assuming there is a CSV file, copy it to the data folder
for f in files:
    if f.endswith('.csv'):
        src = os.path.join(path, f)
        dst = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_tickets.csv')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        print(f"Copied {f} to {dst}")
        
        # Read the first few lines to understand structure
        import pandas as pd
        df = pd.read_csv(dst)
        print("\nColumns:", df.columns.tolist())
        print("\nFirst row:\n", df.iloc[0].to_dict())
        break
