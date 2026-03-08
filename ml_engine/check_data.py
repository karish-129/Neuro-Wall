import pandas as pd

# Load the data
df = pd.read_csv("../data/mixed_security_dataset.csv")

# Get one example of an attack and one example of safe text
print("🔍 DATA INSPECTION")
print("-" * 30)

# Find rows labeled '1'
print("rows labeled '1' (Usually Attack):")
print(df[df['label'] == 1]['text'].head(3).values)

print("\n" + "-" * 30)

# Find rows labeled '0'
print("rows labeled '0' (Usually Safe):")
print(df[df['label'] == 0]['text'].head(3).values)