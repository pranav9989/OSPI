import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from predict import preprocess_input

# 🟡 Streamlit config
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

model_accuracies = {
    "Gradient Boosting": 0.913, 
    "Random Forest": 0.905,
    "Logistic Regression": 0.857,
    "AdaBoost": 0.884,
    "XGBoost": 0.919
}

selected_model = st.sidebar.selectbox("Choose a model", list(model_options.keys()))
model_path = model_options[selected_model]
model = joblib.load(model_path)

# Load test and training data
X_test = joblib.load("model/X_test.pkl")
y_test = joblib.load("model/y_test.pkl")

# Use this line if X_train is available and has column names
# X_train = joblib.load("model/X_train.pkl")
# feature_names = X_train.columns

# Otherwise, manually assign the feature names
feature_names = [
    'Administrative', 'Administrative_Duration', 'ProductRelated',
    'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
    'TrafficType', 'Weekend', 'VisitorType_Other',
    'VisitorType_Returning_Visitor', 'Month_sin', 'Month_cos',
    'SpecialDay_1', 'OperatingSystems_2', 'OperatingSystems_3',
    'OperatingSystems_other', 'Browser_2', 'Browser_other', 'Region_2',
    'Region_3', 'Region_4', 'Region_5', 'Region_6', 'Region_7', 'Region_8',
    'Region_9'
]

# 🔍 Display model accuracy
st.sidebar.markdown(f"**Accuracy**: `{model_accuracies[selected_model]*100:.2f}%`")

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
        OperatingSystems = int(OperatingSystems) if OperatingSystems != 'other' else OperatingSystems
        Browser = int(Browser) if Browser != 'other' else Browser

        user_input_dict = {
            "Administrative": Administrative,
            "Administrative_Duration": Administrative_Duration,
            "ProductRelated": ProductRelated,
            "ProductRelated_Duration": ProductRelated_Duration,
            "BounceRates": BounceRates,
            "ExitRates": ExitRates,
            "PageValues": PageValues,
            "Month": int(Month),
            "SpecialDay": int(SpecialDay),
            "Weekend": int(Weekend),
            "VisitorType": VisitorType,
            "TrafficType": int(TrafficType),
            "OperatingSystems": OperatingSystems,
            "Browser": Browser,
            "Region": int(Region),
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

        # --- Model Evaluation ---
        st.subheader("📊 Model Evaluation on Test Set")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        st.text("🔹 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.text("🔹 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        st.text(f"🔹 AUC-ROC Score: {roc_auc:.4f}")

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic')
        ax2.legend(loc="lower right")
        st.pyplot(fig2)

        # 🔍 Feature Importance
        if hasattr(model, "feature_importances_"):
            st.subheader("🔍 Feature Importances")
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            fig3, ax3 = plt.subplots(figsize=(8, 5))
            sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax3)
            ax3.set_title("Top Feature Importances")
            st.pyplot(fig3)
