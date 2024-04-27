import pandas as pd

# Load the whole dataset
whole_dataset = pd.read_csv('heart.csv')  # Change the file path accordingly

# Drop the specified columns
columns_to_drop = ['HeartDisease', 'RestingBP', 'RestingECG']
whole_dataset = whole_dataset.drop(columns=columns_to_drop)

# Save the modified dataset as 'heartdisease.csv'
whole_dataset.to_csv('heartdisease.csv', index=False)

print("Dataset after dropping columns:")
print(whole_dataset)
