#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


df = pd.read_csv('Generated_Small_Earners_Transactions.csv')
df.head()


# In[7]:


# Selecting relevant columns
X = df[['Total_Earnings', 'Income_StdDev']]
Y = np.where(df['Total_Earnings'] > 50000, 1, 0)  # Eligibility: Total_Earnings must be greater than 50,000

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train RandomForest Model
model = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=42)
model.fit(X_train, Y_train)

# Test Model
predictions = model.predict(X_test)
accuracy = metrics.accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy)


# In[8]:


# Confusion Matrix
conf_matrix = metrics.confusion_matrix(Y_test, predictions)
sn.heatmap(conf_matrix, annot=True, cmap='coolwarm', fmt='d', xticklabels=['Not Eligible', 'Eligible'], yticklabels=['Not Eligible', 'Eligible'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[9]:


filtered_df = df[[ 'Business Type', 'Total_Earnings', 'Income_StdDev']]
print(filtered_df)


# In[11]:


# Deploy Model
Business_Type = input("Enter Business Type: ")
Total_Earnings = float(input("Enter Total Earnings: "))
Income_StdDev = float(input("Enter Income StdDev: "))
pred_new = model.predict(scaler.transform([[Total_Earnings, Income_StdDev]]))
print(f"Business Type: {Business_Type}, Loan Eligibility: {'Eligible' if pred_new[0] == 1 else 'Not Eligible'}")


# In[ ]:




