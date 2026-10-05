import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Credit Risk Prediction System",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# LOAD TRAINED MODEL
# =========================================================
MODEL_PATH = "models/best_model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# =========================================================
# TITLE
# =========================================================
st.title("💳 Credit Risk Prediction System")

st.write(
    "Enter customer and loan information to predict the credit risk."
)

st.divider()

# =========================================================
# CUSTOMER INFORMATION
# =========================================================
st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

with col2:
    residence_since = st.selectbox(
        "Years at Current Residence",
        [1, 2, 3, 4, 5]
    )

with col3:
    personal_status = st.selectbox(
        "Personal Status",
        [
            "Male - divorced/separated",
            "Female - divorced/separated/married",
            "Male - single",
            "Male - married/widowed",
            "Female - single"
        ]
    )

# =========================================================
# BANKING AND CREDIT INFORMATION
# =========================================================
st.subheader("🏦 Banking & Credit Information")

col1, col2 = st.columns(2)

with col1:

    checking_status = st.selectbox(
        "Checking Account Status",
        [
            "Below 0 / Overdrawn",
            "0 to less than 200",
            "200 or more",
            "No checking account"
        ]
    )

    credit_history = st.selectbox(
        "Credit History",
        [
            "No previous credit / all paid properly",
            "Previous credit at this bank paid properly",
            "Existing credits paid properly",
            "Payment delays in past",
            "Problematic credit history"
        ]
    )

    savings_status = st.selectbox(
        "Savings Account Status",
        [
            "Less than 100",
            "100 to less than 500",
            "500 to less than 1000",
            "1000 or more",
            "Unknown / No savings account"
        ]
    )

with col2:

    existing_credits = st.number_input(
        "Number of Existing Credits",
        min_value=0,
        max_value=10,
        value=1
    )

    installment_rate = st.selectbox(
        "Installment Rate",
        [1, 2, 3, 4]
    )

    employment = st.selectbox(
        "Employment Duration",
        [
            "Unemployed",
            "Less than 1 year",
            "1 to 4 years",
            "4 to 7 years",
            "7 years or more"
        ]
    )

# =========================================================
# LOAN INFORMATION
# =========================================================
st.subheader("💰 Loan Information")

col1, col2, col3 = st.columns(3)

with col1:

    duration = st.number_input(
        "Loan Duration (Months)",
        min_value=4,
        max_value=72,
        value=24
    )

with col2:

    credit_amount = st.number_input(
        "Credit Amount",
        min_value=100,
        max_value=100000,
        value=5000
    )

with col3:

    purpose = st.selectbox(
        "Loan Purpose",
        [
            "New car",
            "Used car",
            "Furniture / equipment",
            "Radio / television",
            "Domestic appliances",
            "Repairs",
            "Education",
            "Retraining",
            "Business",
            "Other"
        ]
    )

# =========================================================
# PROPERTY AND HOUSING
# =========================================================
st.subheader("🏠 Property & Housing")

col1, col2 = st.columns(2)

with col1:

    property_magnitude = st.selectbox(
        "Property Type",
        [
            "Real estate",
            "Building society savings agreement / life insurance",
            "Car or other property",
            "Unknown / No property"
        ]
    )

with col2:

    housing = st.selectbox(
        "Housing",
        [
            "Rented",
            "Own house",
            "Free housing"
        ]
    )

# =========================================================
# OTHER FINANCIAL INFORMATION
# =========================================================
st.subheader("📄 Other Financial Information")

col1, col2, col3 = st.columns(3)

with col1:

    other_debtors = st.selectbox(
        "Other Debtors / Guarantors",
        [
            "None",
            "Co-applicant",
            "Guarantor"
        ]
    )

with col2:

    other_payment_plans = st.selectbox(
        "Other Payment Plans",
        [
            "Bank",
            "Stores",
            "None"
        ]
    )

with col3:

    own_telephone = st.selectbox(
        "Telephone",
        [
            "No telephone",
            "Telephone registered in own name"
        ]
    )

# =========================================================
# EMPLOYMENT & PERSONAL DETAILS
# =========================================================
st.subheader("💼 Employment & Personal Details")

col1, col2, col3 = st.columns(3)

with col1:

    job = st.selectbox(
        "Job Type",
        [
            "Unemployed / unskilled non-resident",
            "Unskilled resident",
            "Skilled employee / official",
            "Manager / self-employed / highly qualified / officer"
        ]
    )

with col2:

    num_dependents = st.number_input(
        "Number of Dependents",
        min_value=1,
        max_value=10,
        value=1
    )

with col3:

    foreign_worker = st.selectbox(
        "Foreign Worker",
        [
            "Yes",
            "No"
        ]
    )

st.divider()

# =========================================================
# HUMAN-READABLE VALUES -> ORIGINAL DATASET CODES
# =========================================================

checking_map = {
    "Below 0 / Overdrawn": "A11",
    "0 to less than 200": "A12",
    "200 or more": "A13",
    "No checking account": "A14"
}

credit_history_map = {
    "No previous credit / all paid properly": "A30",
    "Previous credit at this bank paid properly": "A31",
    "Existing credits paid properly": "A32",
    "Payment delays in past": "A33",
    "Problematic credit history": "A34"
}

purpose_map = {
    "New car": "A40",
    "Used car": "A41",
    "Furniture / equipment": "A42",
    "Radio / television": "A43",
    "Domestic appliances": "A44",
    "Repairs": "A45",
    "Education": "A46",
    "Retraining": "A48",
    "Business": "A49",
    "Other": "A410"
}

savings_map = {
    "Less than 100": "A61",
    "100 to less than 500": "A62",
    "500 to less than 1000": "A63",
    "1000 or more": "A64",
    "Unknown / No savings account": "A65"
}

employment_map = {
    "Unemployed": "A71",
    "Less than 1 year": "A72",
    "1 to 4 years": "A73",
    "4 to 7 years": "A74",
    "7 years or more": "A75"
}

personal_status_map = {
    "Male - divorced/separated": "A91",
    "Female - divorced/separated/married": "A92",
    "Male - single": "A93",
    "Male - married/widowed": "A94",
    "Female - single": "A95"
}

other_debtors_map = {
    "None": "A101",
    "Co-applicant": "A102",
    "Guarantor": "A103"
}

property_map = {
    "Real estate": "A121",
    "Building society savings agreement / life insurance": "A122",
    "Car or other property": "A123",
    "Unknown / No property": "A124"
}

payment_plan_map = {
    "Bank": "A141",
    "Stores": "A142",
    "None": "A143"
}

housing_map = {
    "Rented": "A151",
    "Own house": "A152",
    "Free housing": "A153"
}

job_map = {
    "Unemployed / unskilled non-resident": "A171",
    "Unskilled resident": "A172",
    "Skilled employee / official": "A173",
    "Manager / self-employed / highly qualified / officer": "A174"
}

telephone_map = {
    "No telephone": "A191",
    "Telephone registered in own name": "A192"
}

foreign_worker_map = {
    "Yes": "A201",
    "No": "A202"
}

# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "🔍 Predict Credit Risk",
    use_container_width=True
):

    # -----------------------------------------------------
    # CREATE INPUT DATAFRAME
    # IMPORTANT:
    # Column names MUST exactly match training data
    # -----------------------------------------------------

    input_data = pd.DataFrame([{

        "checking_status": checking_map[checking_status],

        "duration": duration,

        "credit_history": credit_history_map[credit_history],

        "purpose": purpose_map[purpose],

        "credit_amount": credit_amount,

        "savings_status": savings_map[savings_status],

        "employment": employment_map[employment],

        "installment_rate": installment_rate,

        "personal_status": personal_status_map[personal_status],

        "other_debtors": other_debtors_map[other_debtors],

        "residence_since": residence_since,

        "property_magnitude": property_map[property_magnitude],

        "age": age,

        "other_payment_plans": payment_plan_map[other_payment_plans],

        "housing": housing_map[housing],

        "existing_credits": existing_credits,

        "job": job_map[job],

        "num_dependents": num_dependents,

        "own_telephone": telephone_map[own_telephone],

        "foreign_worker": foreign_worker_map[foreign_worker]

    }])

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    try:

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        good_probability = probabilities[0] * 100
        bad_probability = probabilities[1] * 100

        st.divider()

        st.subheader("📊 Prediction Result")

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        if prediction == 1:

            st.error("⚠️ BAD CREDIT RISK")

        else:

            st.success("✅ GOOD CREDIT RISK")

        # -------------------------------------------------
        # PROBABILITIES
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Good Credit Probability",
                f"{good_probability:.2f}%"
            )

        with col2:

            st.metric(
                "Bad Credit Probability",
                f"{bad_probability:.2f}%"
            )

        st.write("")

        # -------------------------------------------------
        # PROBABILITY BARS
        # -------------------------------------------------

        st.write("**Good Credit Probability**")

        st.progress(
            min(int(good_probability), 100)
        )

        st.caption(
            f"{good_probability:.2f}%"
        )

        st.write("**Bad Credit Probability**")

        st.progress(
            min(int(bad_probability), 100)
        )

        st.caption(
            f"{bad_probability:.2f}%"
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )