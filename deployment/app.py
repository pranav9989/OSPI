import streamlit as st
import pandas as pd
import joblib
from predict import preprocess_input

# 🟡 This must be the first Streamlit call
st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")

st.title("🛍️ Online Shopper Purchase Intention Predictor")

# 🎛️ Sidebar model selection
st.sidebar.header("⚙️ Model Selector")
model_options = {
    "Gradient Boosting": "model/Gradient_Boosting.pkl",
    "Random Forest": "model/Random_Forest.pkl",
    "Logistic Regression": "model/Logistic_Regression.pkl",
    "AdaBoost": "model/Adaboost.pkl",
    "XGBoost": "model/XGBoost.pkl"
}
selected_model = st.sidebar.selectbox("Choose a model", list(model_options.keys()))
model_path = model_options[selected_model]
model = joblib.load(model_path)

with st.form("prediction_form"):
    st.subheader("Customer Behavior")

    col1, col2 = st.columns(2)
    Administrative = col1.number_input("Administrative", min_value=0, value=1)
    Administrative_Duration = col2.number_input("Administrative Duration (sec)", min_value=0.0, format="%.3f")

    ProductRelated = col1.number_input("Product Related", min_value=0, value=1)
    ProductRelated_Duration = col2.number_input("Product Related Duration (sec)", min_value=0.0, format="%.3f")

    st.markdown("---")
    st.subheader("Analytics Metrics")

    BounceRates = st.number_input("Bounce Rates", min_value=0.0, format="%.6f")
    ExitRates = st.number_input("Exit Rates", min_value=0.0, format="%.6f")
    PageValues = st.number_input("Page Values", min_value=0.0, format="%.6f")

    st.markdown("---")
    st.subheader("Time")

    Month = st.selectbox("Month", list(range(1, 13)))
    SpecialDay = st.selectbox("Special Day", [0, 1])
    Weekend = st.selectbox("Weekend", [0, 1])

    st.markdown("---")
    st.subheader("Visitor Traffic")

    VisitorType = st.radio("Visitor Type", ['Returning_Visitor', 'New_Visitor', 'Other'])
    TrafficType = st.selectbox("Traffic Type", list(range(1, 21)))

    st.markdown("---")
    st.subheader("Additional Info")

    OperatingSystems = st.selectbox("Operating System", [1, 2, 3, 'other'])
    Browser = st.selectbox("Browser", [1, 2, 'other'])
    Region = st.selectbox("Region", list(range(1, 10)))

    submit = st.form_submit_button("Predict")

    if submit:
        # 🛠️ Ensure correct typing
        Month = int(Month)
        SpecialDay = int(SpecialDay)
        Weekend = int(Weekend)
        TrafficType = int(TrafficType)
        Region = int(Region)

        if OperatingSystems != 'other':
            OperatingSystems = int(OperatingSystems)

        if Browser != 'other':
            Browser = int(Browser)

        user_input_dict = {
            "Administrative": Administrative,
            "Administrative_Duration": Administrative_Duration,
            "ProductRelated": ProductRelated,
            "ProductRelated_Duration": ProductRelated_Duration,
            "BounceRates": BounceRates,
            "ExitRates": ExitRates,
            "PageValues": PageValues,
            "Month": Month,
            "SpecialDay": SpecialDay,
            "Weekend": Weekend,
            "VisitorType": VisitorType,
            "TrafficType": TrafficType,
            "OperatingSystems": OperatingSystems,
            "Browser": Browser,
            "Region": Region,
        }

        user_input_df = pd.DataFrame([user_input_dict])
        processed_input = preprocess_input(user_input_df)
        prediction = model.predict(processed_input)[0]

        if prediction == 1:
            st.success(f"✅ Using **{selected_model}**, the customer is **likely to make a purchase**.")
        else:
            st.error(f"❌ Using **{selected_model}**, the customer is **unlikely to purchase**.")

        st.write("📥 Raw Input", user_input_df)
        st.write("🧪 Transformed Input for Model", processed_input)
