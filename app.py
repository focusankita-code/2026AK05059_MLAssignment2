
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Classification")
st.write(
    "This application evaluates five Machine Learning "
    "classification models using the Dry Bean Dataset."
)

# --------------------------------------------------
# 2. Model file paths
# --------------------------------------------------

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "saved_models"
)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "SVM": "svm.joblib"
}

# --------------------------------------------------
# 3. Load saved model
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# --------------------------------------------------
# 4. Select model
# --------------------------------------------------

selected_model = st.selectbox(
    "Select a Machine Learning Model",
    list(MODEL_FILES.keys())
)

model_path = os.path.join(
    MODEL_DIR,
    MODEL_FILES[selected_model]
)

if not os.path.exists(model_path):
    st.error("Model file not found: " + model_path)
    st.stop()

try:
    model = load_model(model_path)
    st.success(selected_model + " model loaded successfully!")

except Exception as error:
    st.error("Error loading model: " + str(error))
    st.stop()


# --------------------------------------------------
# 5. Upload test dataset
# --------------------------------------------------

st.subheader("Upload Test Dataset")

st.write(
    "Upload a labelled CSV file containing the 16 input "
    "features and the target column named Class."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.write("Uploaded Dataset Preview")
        st.dataframe(data.head())

        # --------------------------------------------------
        # 6. Validate dataset
        # --------------------------------------------------

        if "Class" not in data.columns:
            st.error("The uploaded CSV must contain a 'Class' column.")
            st.stop()

        if data.empty:
            st.error("The uploaded dataset is empty.")
            st.stop()

        X = data.drop(columns=["Class"])
        y = data["Class"]

        if y.isnull().any():
            st.error("The target column contains missing values.")
            st.stop()

        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)

            missing_features = [
                feature for feature in expected_features
                if feature not in X.columns
            ]

            extra_features = [
                feature for feature in X.columns
                if feature not in expected_features
            ]

            if missing_features or extra_features:
                st.error(
                    "The uploaded CSV does not have the expected "
                    "feature columns.\n\n"
                    "Missing columns: " + str(missing_features) +
                    "\n\nExtra columns: " + str(extra_features)
                )
                st.stop()

            X = X[expected_features]

        if X.isnull().any().any():
            st.error("Input features contain missing values.")
            st.stop()

        if not all(
            pd.api.types.is_numeric_dtype(X[column])
            for column in X.columns
        ):
            st.error("All input features must be numeric.")
            st.stop()

        if not np.isfinite(X.to_numpy(dtype=float)).all():
            st.error("Input features contain infinite or invalid values.")
            st.stop()

        # --------------------------------------------------
        # 7. Make predictions
        # --------------------------------------------------

        predictions = model.predict(X)

        # --------------------------------------------------
        # 8. Calculate evaluation metrics
        # --------------------------------------------------

        accuracy = accuracy_score(y, predictions)

        precision = precision_score(
            y, predictions,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y, predictions,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y, predictions,
            average="macro",
            zero_division=0
        )

        mcc = matthews_corrcoef(y, predictions)

        try:
            probabilities = model.predict_proba(X)

            auc = roc_auc_score(
                y,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=model.classes_
            )

        except Exception:
            auc = None

        # --------------------------------------------------
        # 9. Display evaluation metrics
        # --------------------------------------------------

        st.subheader("Model Evaluation Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("F1 Score", f"{f1:.4f}")

        if auc is not None:
            col5.metric("AUC", f"{auc:.4f}")
        else:
            col5.metric("AUC", "Not available")

        col6.metric("MCC", f"{mcc:.4f}")

        # --------------------------------------------------
        # 10. Confusion Matrix
        # --------------------------------------------------

        st.subheader("Confusion Matrix")

        labels = model.classes_

        cm = confusion_matrix(
            y,
            predictions,
            labels=labels
        )

        cm_df = pd.DataFrame(
            cm,
            index=labels,
            columns=labels
        )

        st.dataframe(cm_df)

        # --------------------------------------------------
        # 11. Classification Report
        # --------------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y,
            predictions,
            zero_division=0
        )

        st.text(report)

        # --------------------------------------------------
        # 12. Download predictions
        # --------------------------------------------------

        st.subheader("Download Predictions")

        results = data.copy()
        results["Predicted_Class"] = predictions

        csv_output = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv_output,
            file_name="dry_bean_predictions.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error("Something went wrong: " + str(error))

else:
    st.info("Please upload the test_data.csv file to evaluate the model.")
