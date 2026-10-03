import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Prediction System",
    page_icon="💳",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

MODEL_PATH = "models/best_model.joblib"

model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("💳 Credit Risk Prediction System")
st.write(
    "Enter customer financial information to predict credit risk."
)

st.divider()


# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)


with col1:
    checking_status = st.selectbox(
        "Checking Account Status",
        ["A11", "A12", "A13", "A14"]
    )

    duration = st.number_input(
        "Loan Duration (months)",
        min_value=4,
        max_value=72,
        value=24
    )

    credit_history = st.selectbox(
        "Credit History",
        ["A30", "A31", "A32", "A33", "A34"]
    )

    purpose = st.selectbox(
        "Loan Purpose",
        [
            "A40", "A41", "A410", "A42",
            "A43", "A44", "A45", "A46",
            "A48", "A49"
        ]
    )

    credit_amount = st.number_input(
        "Credit Amount",
        min_value=250,
        max_value=20000,
        value=2500
    )

    savings_status = st.selectbox(
        "Savings Account",
        ["A61", "A62", "A63", "A64", "A65"]
    )

    employment = st.selectbox(
        "Employment",
        ["A71", "A72", "A73", "A74", "A75"]
    )


with col2:
    installment_rate = st.number_input(
        "Installment Rate (%)",
        min_value=1,
        max_value=4,
        value=2
    )

    personal_status = st.selectbox(
        "Personal Status",
        ["A91", "A92", "A93", "A94", "A95"]
    )

    other_debtors = st.selectbox(
        "Other Debtors / Guarantors",
        ["A101", "A102", "A103"]
    )

    residence_since = st.number_input(
        "Years at Current Residence",
        min_value=1,
        max_value=4,
        value=2
    )

    property_magnitude = st.selectbox(
        "Property",
        ["A121", "A122", "A123", "A124"]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    other_payment_plans = st.selectbox(
        "Other Payment Plans",
        ["A141", "A142", "A143"]
    )


with col3:
    housing = st.selectbox(
        "Housing",
        ["A151", "A152", "A153"]
    )

    existing_credits = st.number_input(
        "Number of Existing Credits",
        min_value=1,
        max_value=4,
        value=1
    )

    job = st.selectbox(
        "Job Type",
        ["A171", "A172", "A173", "A174"]
    )

    num_dependents = st.number_input(
        "Number of Dependents",
        min_value=1,
        max_value=2,
        value=1
    )

    own_telephone = st.selectbox(
        "Telephone",
        ["A191", "A192"]
    )

    foreign_worker = st.selectbox(
        "Foreign Worker",
        ["A201", "A202"]
    )


st.divider()


# ---------------------------------------------------------
# CREATE INPUT DATAFRAME
# ---------------------------------------------------------

input_data = pd.DataFrame({
    "checking_status": [checking_status],
    "duration": [duration],
    "credit_history": [credit_history],
    "purpose": [purpose],
    "credit_amount": [credit_amount],
    "savings_status": [savings_status],
    "employment": [employment],
    "installment_rate": [installment_rate],
    "personal_status": [personal_status],
    "other_debtors": [other_debtors],
    "residence_since": [residence_since],
    "property_magnitude": [property_magnitude],
    "age": [age],
    "other_payment_plans": [other_payment_plans],
    "housing": [housing],
    "existing_credits": [existing_credits],
    "job": [job],
    "num_dependents": [num_dependents],
    "own_telephone": [own_telephone],
    "foreign_worker": [foreign_worker]
})


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if st.button("🔍 Predict Credit Risk", type="primary"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    bad_probability = probability[1]
    good_probability = probability[0]


    st.divider()

    st.subheader("Prediction Result")


    if prediction == 0:

        st.success("✅ GOOD CREDIT RISK")

        st.write(
            f"Probability of Good Credit: "
            f"**{good_probability * 100:.2f}%**"
        )

        st.write(
            f"Probability of Bad Credit: "
            f"**{bad_probability * 100:.2f}%**"
        )

    else:

        st.error("⚠️ BAD CREDIT RISK")

        st.write(
            f"Probability of Good Credit: "
            f"**{good_probability * 100:.2f}%**"
        )

        st.write(
            f"Probability of Bad Credit: "
            f"**{bad_probability * 100:.2f}%**"
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Credit Risk Prediction System | "
    "Machine Learning + Streamlit + MLOps"
)