import pandas as pd
import numpy as np
from pathlib import Path

# Load the original/reference dataset
reference_data = pd.read_csv("data/german_credit.csv")

# Create a simulated current dataset
current_data = reference_data.sample(
    n=200,
    random_state=42
).copy()

# Make a few realistic changes to simulate data drift
np.random.seed(42)

# Customers are slightly older
current_data["age"] = np.clip(
    current_data["age"] + np.random.randint(5, 16, size=len(current_data)),
    18,
    100
)

# Credit amounts are slightly higher
current_data["credit_amount"] = (
    current_data["credit_amount"] * 1.25
).round()

# Loan duration is slightly longer
current_data["duration"] = np.clip(
    current_data["duration"] + 6,
    4,
    72
)

# Create data folder if needed
Path("data").mkdir(exist_ok=True)

# Save simulated current data
current_data.to_csv(
    "data/current_data.csv",
    index=False
)

print("=" * 60)
print("CURRENT DATA CREATED SUCCESSFULLY")
print("=" * 60)

print(f"\nRows: {len(current_data)}")
print(f"Columns: {len(current_data.columns)}")

print("\nSaved to:")
print("data/current_data.csv")