import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from predict import preprocess_input

# 🛍️ Streamlit Configuration
st.set_page_config(page_title="🎩 Shopper Purchase Predictor", layout="wide", page_icon="🛒")

# 🌎 Multi-page app support
page = st.sidebar.radio("Navigate", ["Predictor"])

if page == "Predictor":
    st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🛍️ Online Shopper Purchase Intention Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # 🎛️ Sidebar - Model Selection
    with st.sidebar:
        st.header("⚙️ Model Settings")
        st.markdown("Select a classification model to use:")

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

        selected_model = st.selectbox("🧠 Choose Model", list(model_options.keys()))
        st.markdown(f"🔍 **Accuracy**: `{model_accuracies[selected_model]*100:.2f}%`")
        model = joblib.load(model_options[selected_model])

    # 🔢 Load Data for Evaluation
    X_test = joblib.load("model/X_test.pkl")
    y_test = joblib.load("model/y_test.pkl")

    # 📜 Feature Names
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

    # 🧾 Input Form
    with st.form("prediction_form"):
        st.subheader("📝 Enter Shopper Info")
        col1, col2 = st.columns(2)

        with col1:
            Administrative = st.number_input("📄 Administrative", min_value=0, value=1)
            ProductRelated = st.number_input("🛒 Product Related", min_value=0, value=1)
            BounceRates = st.number_input("↩️ Bounce Rates", min_value=0.0, format="%.6f")
            PageValues = st.number_input("📈 Page Values", min_value=0.0, format="%.6f")
            SpecialDay = st.selectbox("🎉 Special Day", [0, 1])
            VisitorType = st.radio("🧑‍💻 Visitor Type", ['Returning_Visitor', 'New_Visitor', 'Other'])
            OperatingSystems = st.selectbox("💻 Operating System", [1, 2, 3, 'other'])
            Region = st.selectbox("🌍 Region", list(range(1, 10)))

        with col2:
            Administrative_Duration = st.number_input("🕒 Administrative Duration (sec)", min_value=0.0, format="%.3f")
            ProductRelated_Duration = st.number_input("⏱ Product Related Duration (sec)", min_value=0.0, format="%.3f")
            ExitRates = st.number_input("🚪 Exit Rates", min_value=0.0, format="%.6f")
            Month = st.selectbox("📆 Month (1–12)", list(range(1, 13)))
            Weekend = st.selectbox("🗓 Weekend", [0, 1])
            TrafficType = st.selectbox("🚦 Traffic Type", list(range(1, 21)))
            Browser = st.selectbox("🌐 Browser", [1, 2, 'other'])

        submit = st.form_submit_button("📊 Predict")

    # 🎯 Prediction
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

        result_box_color = "#black" if prediction == 1 else "black"
        result_text = "✅ Customer is **likely to purchase**." if prediction == 1 else "❌ Customer is **unlikely to purchase**."
        st.markdown(
            f"<div style='background-color: {result_box_color}; padding: 16px; border-radius: 10px;'>{result_text}</div>",
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.expander("📅 Raw Input").write(user_input_df)
        st.expander("🧪 Transformed Input for Model").write(processed_input)

        st.subheader("📈 Model Evaluation on Test Set")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        col1, col2 = st.columns(2)
        with col1:
            st.text("🔹 Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.text("🔹 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        with col2:
            roc_auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            st.text(f"🔹 AUC-ROC Score: {roc_auc:.4f}")
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC Curve')
            ax2.legend(loc="lower right")
            st.pyplot(fig2)

        if hasattr(model, "feature_importances_"):
            st.subheader("🔍 Feature Importances")
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            fig3, ax3 = plt.subplots(figsize=(8, 5))
            sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax3, palette="crest")
            ax3.set_title("Top Feature Importances")
            st.pyplot(fig3)

elif page == "Dashboard":
    st.title("🌐 Analytics Dashboard")
    df = joblib.load("model/test_dataframe.pkl")  # Original data before splitting

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Sessions", df.shape[0])
        st.metric("Purchase Rate", f"{(df['Revenue'].mean()*100):.2f}%")

    with col2:
        st.metric("Average Page Values", f"{df['PageValues'].mean():.2f}")
        st.metric("Bounce Rate", f"{df['BounceRates'].mean()*100:.2f}%")

    st.markdown("---")
    st.subheader("Purchases by Traffic Type")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='TrafficType', hue='Revenue', ax=ax1, palette='pastel')
    st.pyplot(fig1)

    st.subheader("Purchases by Month")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x='Month', y='PageValues', hue='Revenue', ax=ax2)
    st.pyplot(fig2)

    st.subheader("Visitor Type vs Purchase")
    fig3, ax3 = plt.subplots()
    sns.countplot(data=df, x='VisitorType', hue='Revenue', ax=ax3, palette='Set2')
    st.pyplot(fig3)