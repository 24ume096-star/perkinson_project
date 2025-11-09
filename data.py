import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("parkinsons.data")  # Path to your downloaded CSV

# Drop the 'name' column
df = df.drop(columns=["name"])

# Split into train (80%) and unseen test (20%)
train_df, unseen_test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['status']
)

# Save unseen test set for evaluation
unseen_test_df.to_csv("parkinsons_unseen_test.csv", index=False)
print("✅ Unseen test set saved as 'parkinsons_unseen_test.csv'")
