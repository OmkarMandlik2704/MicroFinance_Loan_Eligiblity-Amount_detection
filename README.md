# MicroFinance_Loan_Eligiblity-Amount_detection
This project simplifies microfinance for small earners by removing CIBIL scores, credit cards, and heavy documentation. ML analyzes 52 weeks of transactions, focusing on income consistency. Random Forest predicts loan eligibility, while Linear Regression estimates loan amounts based on earnings and income variability.

Detailed Description.

Overview
This project aims to simplify access to microfinance for small earners such as milk farmers and street vendors by eliminating traditional credit assessments like CIBIL scores, credit card history, and heavy documentation requirements. Instead, the system uses machine learning (ML) to analyze past financial transactions over 52 weeks, focusing on income consistency to determine loan eligibility and loan amount prediction.

Key Features
Eliminating Barriers

Traditional microfinance institutions rely on credit scores, making it difficult for small earners with no formal financial history to secure loans.
Our model removes the need for CIBIL scores and credit card records, making microfinance more inclusive.
Transaction-Based Eligibility Analysis

The system uses past 52 weeks of transaction data to analyze income patterns, considering total earnings and income variability.
Consistency in earnings is a key factor in determining loan eligibility.
Machine Learning Implementation

Loan Eligibility Prediction (Classification Model)

Uses a Random Forest Classifier to classify applicants as eligible or not eligible based on income consistency.
Features Used:
Total earnings
Income standard deviation (Income_StdDev)
Eligibility Criterion: If total earnings exceed ₹50,000, the applicant is classified as eligible.
Model Performance: Evaluated using a confusion matrix and accuracy score.
Loan Amount Prediction (Regression Model)

Uses Linear Regression to predict the loan amount based on earnings and income variability.
Formula:
𝐿
𝑜
𝑎
𝑛
𝐴
𝑚
𝑜
𝑢
𝑛
𝑡
=
(
𝑇
𝑜
𝑡
𝑎
𝑙
𝐸
𝑎
𝑟
𝑛
𝑖
𝑛
𝑔
𝑠
×
0.6
)
−
(
𝐼
𝑛
𝑐
𝑜
𝑚
𝑒
_
𝑆
𝑡
𝑑
𝐷
𝑒
𝑣
×
10
)
LoanAmount=(TotalEarnings×0.6)−(Income_StdDev×10)
Adjusted thresholds ensure fair loan allocation:
Earnings < ₹15,000 → Max Loan: ₹3,000
₹15,000 - ₹30,000 → Max Loan: ₹6,000
₹30,000 - ₹50,000 → Max Loan: ₹10,000
₹50,000 - ₹1,00,000 → 20% of earnings
Earnings > ₹1,00,000 → Max Loan: ₹25,000
User Interaction

The system takes user input for business type, total earnings, and income variability to determine loan eligibility and predict the possible loan amount.
A visual analysis includes bar charts and threshold indicators, showing loan amounts relative to earnings.
Impact and Applications
Financial Inclusion: Helps small earners get loans without requiring complex documentation.
Data-Driven Decision Making: ML-based analysis ensures fair loan allocation based on past financial behavior.
Scalability: The model can be expanded to include more financial behavior factors and optimize risk assessment further.
This project bridges the financial gap for small business owners and laborers, offering an innovative data-driven approach to microfinance.
