#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore")

import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_csv('Small_Earners_Transactions_Final.csv')

# ------------------------------
# Target Variable with Randomized Threshold Logic
# ------------------------------
np.random.seed(42)
threshold = 50000
noise = np.random.normal(0, 5000, size=len(df))  # stddev=5000
adjusted_earnings = df['Total_Earnings'] + noise
Y = np.where(adjusted_earnings > threshold, 1, 0)

# ------------------------------
# Features
# ------------------------------
X = df[['Total_Earnings', 'Income_StdDev']]

# ------------------------------
# Train-test split
# ------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ------------------------------
# Standardize features
# ------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# Model - Random Forest (depth limited to prevent overfitting)
# ------------------------------
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
model.fit(X_train, Y_train)

# ------------------------------
# Prediction & Evaluation
# ------------------------------
predictions = model.predict(X_test)
accuracy = metrics.accuracy_score(Y_test, predictions)
f1 = metrics.f1_score(Y_test, predictions)
recall = metrics.recall_score(Y_test, predictions)
precision = metrics.precision_score(Y_test, predictions)
auc = metrics.roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])

print("\nModel Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC Score: {auc:.4f}")

# ------------------------------
# Confusion Matrix
# ------------------------------
conf_matrix = metrics.confusion_matrix(Y_test, predictions)
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
labels = np.array([["TN\n{}\n{:.1%}".format(conf_matrix[0, 0], conf_matrix_norm[0, 0]),
                    "FP\n{}\n{:.1%}".format(conf_matrix[0, 1], conf_matrix_norm[0, 1])],
                   ["FN\n{}\n{:.1%}".format(conf_matrix[1, 0], conf_matrix_norm[1, 0]),
                    "TP\n{}\n{:.1%}".format(conf_matrix[1, 1], conf_matrix_norm[1, 1])]])

plt.figure(figsize=(8, 6))
sn.heatmap(conf_matrix, annot=labels, fmt="", cmap='coolwarm',
           xticklabels=['Not Eligible', 'Eligible'],
           yticklabels=['Not Eligible', 'Eligible'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix with TP/TN/FP/FN & Percentages")
plt.show()

# ------------------------------
# Deployment Input
# ------------------------------
Business_Type = input("\nEnter Business Type: ")
Total_Earnings = float(input("Enter Total Earnings: "))
Income_StdDev = float(input("Enter Income StdDev: "))
PAN_CARD = input("Enter PAN Number: ")
AADHAR_CARD = input("Enter AADHAR Number: ")

pred_new = model.predict(scaler.transform([[Total_Earnings, Income_StdDev]]))
print(f"\nBusiness Type: {Business_Type}")
print(f"PAN: {PAN_CARD}, AADHAR: {AADHAR_CARD}")
print(f"Loan Eligibility: {'Eligible' if pred_new[0] == 1 else 'Not Eligible'}")

# ------------------------------
# Display First Few Rows
# ------------------------------
print("\nFirst Few Rows of Dataset:")
print(df.head())


# In[ ]:




