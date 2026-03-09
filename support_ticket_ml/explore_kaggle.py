import pandas as pd
import json

df = pd.read_csv('data/raw_tickets.csv')
info = {
    'columns': df.columns.tolist(),
    'row1': df.iloc[0].fillna('NaN').to_dict()
}

with open('data/kaggle_format.json', 'w') as f:
    json.dump(info, f, indent=2)
