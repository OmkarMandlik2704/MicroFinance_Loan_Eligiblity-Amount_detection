#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv('Generated_Small_Earners_Transactions.csv')
df.head()

# Create Loan Amount column with adjusted threshold (60% of total earnings - income variability)
df['Loan_Amount'] = df['Total_Earnings'] * 0.6 - (df['Income_StdDev'] * 10)  # Increased threshold to 60%

# Define features (X) and target variable (Y)
X = df[['Total_Earnings', 'Income_StdDev']]
Y = df['Loan_Amount']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=110)

# Train Model
file1 = LinearRegression()
result = file1.fit(X_train, y_train)

# Test Model
predictions = result.predict(X_test)

# r2 Score
print("R2 Score:", r2_score(y_test, predictions))



# In[3]:


import matplotlib.pyplot as plt
import numpy as np

# Define loan eligibility threshold based on microfinance logic
def microfinance_threshold(earnings):
    if earnings < 15000:
        return 3000
    elif 15000 <= earnings < 30000:
        return 6000
    elif 30000 <= earnings < 50000:
        return 10000
    elif 50000 <= earnings < 100000:
        return 0.2 * earnings  # 20% of earnings
    else:
        return 25000  # Max loan for high earners

# Apply threshold function to dataset
df['Loan_Threshold'] = df['Total_Earnings'].apply(microfinance_threshold)

# Define colors: Red for below threshold, Green for above
colors = ['red' if loan < thresh else 'green' for loan, thresh in zip(df['Loan_Amount'], df['Loan_Threshold'])]

# Plot bar chart
plt.figure(figsize=(12, 6))
plt.bar(df['Total_Earnings'], df['Loan_Amount'], color=colors, label="Loan Amount")

# Add threshold line
plt.axhline(y=df['Loan_Threshold'].mean(), color='blue', linestyle='dashed', linewidth=2, 
            label=f"Avg Threshold: {df['Loan_Threshold'].mean():.2f}")

# Labels and Title
plt.xlabel("Total Earnings")
plt.ylabel("Loan Amount")
plt.title("Microfinance Loan Amount vs. Total Earnings (With Threshold Indicator)")
plt.legend()
plt.xticks(rotation=45)

plt.show()


# In[5]:


# Take user input for Total Earnings and Income Standard Deviation
total_earnings = float(input("Enter Total Earnings: "))
income_stddev = float(input("Enter Income Standard Deviation: "))

# Create input data for prediction
new_input = [[total_earnings, income_stddev]]

# Predict loan amount
pred_new = result.predict(new_input)

# Display the result
print("Predicted Loan Amount:", pred_new[0])

