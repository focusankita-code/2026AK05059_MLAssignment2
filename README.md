# Dry Bean Classification using Machine Learning

## 1. Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to classify dry beans into their respective categories using physical measurements.

Five machine learning algorithms are implemented and compared using standard classification evaluation metrics.

## 2. Dataset Description

**Dataset:** Dry Bean Dataset

**Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

Dataset details:
- Total instances: 13,611
- Number of input features: 16
- Target variable: Class
- Number of classes: 7
- Features include area, perimeter, major axis length, minor axis length, aspect ratio, eccentricity and other physical measurements.

## 3. GitHub Repository

**Repository Link:** https://github.com/focusankita-code/2026AK05059_MLAssignment2

The repository contains:
- Jupyter Notebook with model implementation and evaluation.
- Python Streamlit application (`app.py`).
- Saved trained machine learning models.
- Test dataset (`test_data.csv`).
- Model comparison metrics (`comparison_metrics.csv`).
- Dependencies (`requirements.txt`).

## 4. Machine Learning Models Implemented

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 5. Model Performance Comparison

The models were evaluated using Accuracy, AUC, Precision, Recall, F1-score and Matthews Correlation Coefficient (MCC).

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9948 | 0.9354 | 0.9321 | 0.9335 | 0.9050 |
| Decision Tree | 0.8917 | 0.9448 | 0.9084 | 0.9088 | 0.9085 | 0.8691 |
| KNN | 0.9166 | 0.9833 | 0.9320 | 0.9271 | 0.9293 | 0.8992 |
| Gaussian Naive Bayes | 0.7639 | 0.9672 | 0.7744 | 0.7694 | 0.7677 | 0.7154 |
| SVM | 0.9221 | 0.9951 | 0.9353 | 0.9320 | 0.9335 | 0.9059 |

*Note: Precision, Recall and F1-score are macro averages. AUC is macro one-vs-rest.*

## 6. Observations

### Logistic Regression
- Achieved an accuracy of 92.14%.
- Produced the highest precision and recall among the five models.
- Demonstrated strong classification performance.

### Decision Tree
- Achieved an accuracy of 89.17%.
- Provided reasonable classification performance.
- Its overall performance was lower than Logistic Regression and SVM.

### K-Nearest Neighbors
- Achieved an accuracy of 91.66%.
- Demonstrated good classification performance.
- Performed better than Decision Tree and Gaussian Naive Bayes.

### Gaussian Naive Bayes
- Achieved an accuracy of 76.39%.
- Had the lowest accuracy among the evaluated models.
- Its performance indicates that the independence assumptions may not adequately represent the dataset.

### Support Vector Machine
- Achieved the highest accuracy of 92.21%.
- Achieved the highest AUC of 0.9951 and MCC of 0.9059.
- Demonstrated the best overall performance based on the reported results.

## 7. Overall Conclusion

Among the five evaluated machine learning algorithms, Support Vector Machine (SVM) achieved the highest accuracy, AUC and MCC.

Logistic Regression also performed very well and achieved the highest precision and recall.

Therefore, SVM is selected as the overall best-performing model for this Dry Bean Classification task based on the reported evaluation results.

## 8. Streamlit Application

The Streamlit application allows users to:
- Upload a labelled test CSV file.
- Select one of the five trained machine learning models.
- View classification evaluation metrics.
- View the confusion matrix and classification report.
- Download prediction results.

**Live Streamlit Application:** To be added after deployment.

