import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load models & scalers
rf_model = joblib.load('rf_classifier.pkl')
scaler_clf = joblib.load('scaler_clf.pkl')

# Load dataset
df = pd.read_csv("insurance_claims.csv")

# Initialize session state for approval status
if "approval_status" not in st.session_state:
    st.session_state.approval_status = None

st.title('Medical Insurance Approval and Claim Estimator!')

# Step 1: Insurance Approval Check
st.header('Step 1: Check if your insurance claim is approved')

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
charges = st.number_input("Medical Charges", min_value=0, value=5000)
smoker = st.selectbox("Smoker?", ["No", "Yes"])
diabetes = st.selectbox("Diabetes Condition", ["No", "Controlled", "Uncontrolled"])
cholesterol = st.selectbox("Cholesterol Level", ["Normal", "High"])
hypertension = st.selectbox("Hypertension Condition", ["No", "Controlled", "Uncontrolled"])
exercise = st.selectbox("Do you exercise regularly?", ["No", "Yes"])
policy_type = st.selectbox("Policy Type", ["Basic", "Standard", "Premium"])

# Define policy max payout
max_payout_dict = {"Basic": 30000, "Standard": 50000, "Premium": 75000}
max_payout = max_payout_dict[policy_type]

# Encode categorical values
smoker_encoded = 1 if smoker == "Yes" else 0
diabetes_encoded = 2 if diabetes == "Uncontrolled" else (1 if diabetes == "Controlled" else 0)
cholesterol_encoded = 1 if cholesterol == "High" else 0
hypertension_encoded = 2 if hypertension == "Uncontrolled" else (1 if hypertension == "Controlled" else 0)
exercise_encoded = 1 if exercise == "Yes" else 0

# Create DataFrame
user_data = pd.DataFrame([[age, charges, smoker_encoded, diabetes_encoded, cholesterol_encoded, 
                           hypertension_encoded, exercise_encoded, max_payout]],
                         columns=['age', 'charges', 'smoker_encoded', 'diabetes_encoded', 
                                  'cholesterol_encoded', 'hypertension_encoded', 'exercise_encoded', 'max_payout'])

# Scale the input for classification
user_data_scaled = scaler_clf.transform(user_data) 

# Approval Prediction Button
if st.button("Check Approval"):
    st.session_state.approval_status = rf_model.predict(user_data_scaled)[0]

# Step 2: Estimate Claim Amount (Only if approved)
if st.session_state.approval_status == 1:
    st.success("✅ Your claim has been approved!")
    st.header("Step 2: Estimate Your Claim Amount")
    st.write(f"💰 Your policy's max payout is ₹{max_payout}")

    # Function to calculate received amount
    def calculate_received_amount(row):
        claim_amount = min(row['charges'], row['max_payout'])  # Cap at max payout
        deduction = 0
        if row['smoker_encoded'] == 1:
            deduction += 0.10  # 10% reduction for smokers
        if row['diabetes_encoded'] == 2:
            deduction += 0.10  # 10% reduction for uncontrolled diabetes
        if row['diabetes_encoded'] == 1:
            deduction += 0.05  # 5% reduction for controlled diabetes
        if row['hypertension_encoded'] == 2:
            deduction += 0.10  # 10% reduction for uncontrolled hypertension
        if row['hypertension_encoded'] == 1:
            deduction += 0.05  # 5% reduction for controlled hypertension
        if row['cholesterol_encoded'] == 1:
            deduction += 0.10  # 10% reduction for high cholesterol  
        if row['exercise_encoded'] == 1:
            deduction -= 0.05  # 5% increase if exercising

        received_amount = claim_amount * (1 - deduction)
        return max(received_amount, 0)  # Ensure non-negative value

    # Button for claim estimation
    if st.button("Estimate Claim Amount"):
        received_amount = calculate_received_amount(user_data.iloc[0])
        st.success(f"💰 Estimated Claim Amount: ₹{received_amount:.2f}")

elif st.session_state.approval_status == 0:
    st.error("❌ Your claim was not approved.")

# ===== Data Insights & Analytics =====

st.sidebar.header("📊 Data Insights & Analytics")

# 1. Percentage of claims approved vs. rejected
st.sidebar.subheader("✅ Claim Approval Statistics")
approval_rate = df['recieved_amount'].apply(lambda x: "Approved" if x > 0 else "Rejected").value_counts()
st.sidebar.bar_chart(approval_rate)

# 2. Average Claim Amount per Category
st.sidebar.subheader("💰 Avg. Claim Amount by Category")
avg_claim_per_category = df.groupby('smoker')['recieved_amount'].mean()
st.sidebar.bar_chart(avg_claim_per_category)

# 3. Heatmap of Feature Correlations
st.sidebar.subheader("🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df[['age', 'charges', 'smoker_encoded', 'diabetes_encoded', 
                'cholesterol_encoded', 'hypertension_encoded', 'recieved_amount']].corr(), annot=True, cmap='coolwarm', ax=ax)
st.sidebar.pyplot(fig)

# 4. Pie Chart: Distribution of Approvals Across Demographics
st.sidebar.subheader("🧑‍⚕️ Approvals by Demographics")
fig, ax = plt.subplots()
df['sex'].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, cmap="Pastel1", ax=ax)
st.sidebar.pyplot(fig)

# ✅ 4. Pie Chart: Approvals by Smoking Status
st.sidebar.subheader("🚬 Approvals by Smoking Status")
fig, ax = plt.subplots()
df['smoker'].value_counts().plot.pie(autopct="%1.1f%%", startangle=90, cmap="Pastel1", ax=ax)
st.sidebar.pyplot(fig)

# ✅ 6. Bar Chart: Claim Amount by Diabetes & Hypertension Levels
st.sidebar.subheader("🩺 Claim Amount by Diabetes & Hypertension")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=df, x='diabetes', y='recieved_amount', hue='hypertension', palette="coolwarm", ax=ax)
ax.set_ylabel("Avg. Received Claim Amount")
st.sidebar.pyplot(fig)

# ✅ 7. Box Plot: Claim Distribution by Age Groups
st.sidebar.subheader("🎭 Claim Distribution by Age Group")
df['age_group'] = pd.cut(df['age'], bins=[18, 30, 45, 60, 80, 100], labels=["18-30", "31-45", "46-60", "61-80", "81+"])
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x="age_group", y="recieved_amount", data=df, hue="age_group", palette="Set2", legend=False, ax=ax)
ax.set_ylabel("Claim Amount Distribution")
st.sidebar.pyplot(fig)

# ✅ 9. Stacked Bar Chart: Exercise & Smoking Impact on Approvals
st.sidebar.subheader("🏃‍♂️ Exercise & Smoking Impact on Approvals")
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x='exercise', hue='smoker', palette="viridis", ax=ax)
ax.set_ylabel("Number of Claims")
st.sidebar.pyplot(fig)

# 5. Line Graph: Claim Trends Over Age
st.sidebar.subheader("📈 Claim Amount Trends by Age")
st.sidebar.line_chart(df.groupby('age')['recieved_amount'].mean())

st.sidebar.write("📌 **Insights:**")
st.sidebar.write("- Smokers tend to have lower approved claims.")
st.sidebar.write("- Higher medical charges correlate with lower claim approval.")
st.sidebar.write("- Regular exercise increases the likelihood of approval.")

st.sidebar.success("💡 Use these insights to improve claim approvals!")