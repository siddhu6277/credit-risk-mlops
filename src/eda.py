import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load dataset
df = pd.read_csv("data/german_credit.csv")

print("=" * 60)
print("CREDIT RISK DATASET - EDA")
print("=" * 60)

# Basic information
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Target distribution
print("\nCredit Risk Distribution:")
print(df["credit_risk"].value_counts())

print("\nCredit Risk Percentage:")
print(df["credit_risk"].value_counts(normalize=True) * 100)

# Numerical summary
print("\nNumerical Summary:")
print(df.describe())

# Create reports folder
Path("reports").mkdir(exist_ok=True)

# Target distribution chart
plt.figure(figsize=(7, 5))
sns.countplot(x="credit_risk", data=df)
plt.title("Credit Risk Distribution")
plt.xlabel("Credit Risk (0 = Good, 1 = Bad)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("reports/credit_risk_distribution.png")
plt.show()

# Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("reports/age_distribution.png")
plt.show()

print("\nEDA completed successfully!")
print("Charts saved inside the reports folder.")