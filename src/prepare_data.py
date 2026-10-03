import pandas as pd
from pathlib import Path

# File paths
input_file = Path("data/german.data")
output_file = Path("data/german_credit.csv")

# Column names for the German Credit dataset
columns = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
    "credit_risk"
]

# Read the original dataset
df = pd.read_csv(
    input_file,
    sep=r"\s+",
    header=None,
    names=columns
)

# Convert target:
# 1 = Good credit
# 2 = Bad credit
df["credit_risk"] = df["credit_risk"].map({
    1: 0,
    2: 1
})

# Save as CSV
df.to_csv(output_file, index=False)

# Display basic information
print("Dataset prepared successfully!")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print("\nFirst 5 rows:")
print(df.head())

print("\nTarget distribution:")
print(df["credit_risk"].value_counts())

print(f"\nSaved to: {output_file}")