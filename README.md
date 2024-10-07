Telco Customer Churn RFM Analysis
Project Overview
This project analyzes customer churn for a telecommunications company using the RFM (Recency, Frequency, Monetary) model. By leveraging the customer data, the aim is to segment customers based on their behavior, which can assist in identifying potential churners and enhancing customer retention strategies.

Dataset Description
The dataset used for this analysis is the Telco Customer Churn dataset, which contains information about customers and their subscription details. Key variables include:

customerID: Unique identifier for each customer.
gender: Gender of the customer (Male/Female).
SeniorCitizen: Indicates if the customer is a senior citizen (0: No, 1: Yes).
Partner: Indicates if the customer has a partner (Yes/No).
Dependents: Indicates if the customer has dependents (Yes/No).
tenure: Duration of stay with the company (in months).
PhoneService: Indicates if the customer has phone service (Yes/No).
InternetService: Type of internet service (DSL/Fiber/No).
Contract: Type of contract (Month-to-month/One year/Two year).
TotalCharges: Total amount charged to the customer.
Churn: Indicates if the customer has churned (Yes/No).
Analysis Steps
Data Preparation:

Load the dataset and perform initial data checks for missing values and data types.
Convert the TotalCharges column to numeric format.
RFM Calculation:

Calculate Recency, Frequency, and Monetary values based on customer tenure, contract type, and total charges.
Assign scores to each RFM dimension using quantiles.
Segmentation:

Create customer segments based on the RFM scores using a predefined segmentation map.
Analyze the segments to understand customer behavior.
Results:

Extract and analyze the segments with the highest and lowest scores.
Visualize the distribution of segments and their characteristics.
