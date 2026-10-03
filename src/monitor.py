import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# ---------------------------------------------------------
# LOAD REFERENCE AND CURRENT DATA
# ---------------------------------------------------------

reference_data = pd.read_csv("data/german_credit.csv")

# Use a sample of the same dataset as "current" data
# for our local monitoring demonstration.
current_data = reference_data.sample(
    n=200,
    random_state=42
).copy()

# Remove target from drift analysis.
# We want to monitor the input features.
reference_data = reference_data.drop(
    columns=["credit_risk"]
)

current_data = current_data.drop(
    columns=["credit_risk"]
)

# ---------------------------------------------------------
# CREATE EVIDENTLY REPORT
# ---------------------------------------------------------

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

snapshot = report.run(
    current_data=current_data,
    reference_data=reference_data
)

# ---------------------------------------------------------
# SAVE HTML REPORT
# ---------------------------------------------------------

snapshot.save_html(
    "reports/data_drift_report.html"
)

print("=" * 60)
print("EVIDENTLY MONITORING COMPLETED")
print("=" * 60)
print("\nReference rows :", len(reference_data))
print("Current rows   :", len(current_data))
print("\nReport saved to:")
print("reports/data_drift_report.html")