import pandas as pd
import json

# Read the Excel file
df = pd.read_excel('cricket_train.xlsx')

# Print structure
print("=" * 50)
print("Dataset Information")
print("=" * 50)
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 2 examples:")
print(df.head(2))
print("=" * 50)

# Convert to JSONL format
data = []
for _, row in df.iterrows():
    entry = {
        "instruction": str(row['instruction']),
        "input": str(row['input']),
        "output": str(row['output'])
    }
    data.append(entry)

# Save as JSONL
with open('cricket_train.jsonl', 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')

print(f"\n✓ Saved {len(data)} training examples to cricket_train.jsonl")
