# Bank-Customer-Churn-Prediction

## Introduction
A manager at a local bank is disturbed with more and more customers leaving their credit card services. They need a way of predicting which customers are most likely to stop using their credit card products (Customer Churn) in order to proactively check in on the customer to provide them better services in order to convince them to change their minds. You are given a dataset of 10,000 customers with 18 features per customer. Roughly 16% of the current customer base have churned so far, so it will be difficult to predict the ones who will.

As you analyze the data, before you create the model, the sales team also needs you to determine the most influential factors that can lead to a customer's decision of leaving the business. The head of the sales department is expecting a report that helps them visualize where the differences lie between churning and non-churning customers.

# Business Understanding
To define the success of the solution that we will deliver let's define the metrics as: F1 Score, Precision and Recall. This metrics were chosen since normally churn problems are imbalanced, but all depends on the definition of churn and the cost driven by each scenario.

## Objective
1. Identify which customers are most likely to be churned so the bank manager knows who to provide a better service to
  - The Top Priority is to identify churning customers,as if we predict non-churning customers as churned, it won't harm our business, but predicting churning customers as Non-churning will. False negatives won't hurt us, but False Positives do
  - This task is binary classification
2. A clean and easy to understand visual report that helps the sales team better visualize what makes a client churn or not churn
3. Precision and Recall Curves as well as the Confusion Matrix will also be used

**Performance of the model will be mesured with accuracy and the rate of False Positives. The manager is looking for at least a 90% F1 Score accuracy**

## Issues
since only 16% of the customers are churned, a data upsampling method is needed to match them with the regular customer size to give our model a better chance of catching small details that would be missed had we not upsampled it

# Data Understanding
The Backend Engineer at the bank gives us the data through their MySQL database in an easy to use CSV with all missing features replaced by an "unkown" string. However, he tried to train a Naive Bayes classifier and accidently left in 2 prediction columns in the data. No Worries. We'll also remove the Clientnumber as this isn't important

## Feature Description
- CLIENTNUM: Client number. Unique identifier for the customer holding the account

- Customer_Age: Demographic variable - Customer's Age in Years

- Gender: Demographic variable - M=Male, F=Female

- Dependent_count: Demographic variable - Number of dependents

- Education_Level: Demographic variable - Educational Qualification of the account holder (example: high school, college graduate, etc.)

- Marital_Status: Demographic variable - Married, Single, Divorced, Unknown

- Income_Category: Demographic variable - Annual Income Category of the account holder (<  40K, 40K - 60K,  60K− 80K,  80K− 120K, > $120K, Unknown)

- Card_Category: Product Variable - Type of Card (Blue, Silver, Gold, Platinum)

- Months_on_book: Period of relationship with bank

- Total_Relationship_Count: Total no. of products held by the customer

- Months_Inactive_12_mon: No. of months inactive in the last 12 months

- Contacts_Count_12_mon: No. of Contacts in the last 12 months

- Credit_Limit: Credit Limit on the Credit Card

- Total_Revolving_Bal: Total Revolving Balance on the Credit Card

- Avg_Open_To_Buy: Open to Buy Credit Line (Average of last 12 months)

- Total_Amt_Chng_Q4_Q1: Change in Transaction Amount (Q4 over Q1)

- Total_Trans_Amt: Total Transaction Amount (Last 12 months)

- Total_Trans_Ct: Total Transaction Count (Last 12 months)

- Total_Ct_Chng_Q4_Q1: Change in Transaction Count (Q4 over Q1)

- Avg_Utilization_Ratio: Average Card Utilization Ratio

## Target
Attrition_Flag: Internal event (customer activity) variable - if the account is closed then 1(Attrited Customer) else 0(Existing Customer)

# Exploring the Data
After i'm exploring and visualize the data, we can conclude : 

- The customer gender is almost even, 30% college graduates with half being either Highschool graduates, unkown, or uneducated. The remaining 40% are either current college students,or grad students.
- Almost half are married, 38% single, and the remaining 12 are divorced or unkown. 35% of customers make less than $40k per year which is near the poverty threshold. The rest are more evenly spaced out. 93% of customers choose the cheapest card option (likely the lowest interest rate) with a tiny portion choosing the more expensive cards
- Churned customers are likely to hold less credit cards than existing customers which is shown by a lower median . Is there a deal you provide that favors customers with multiple credit cards? (Like customers with spouses, families, or buisnesses that need additional cards)
- Churned customers tend to have slightly more inactive months, but the distribution is more concentrated from the 1-4 months inactive (though this may be from the small sample size)
- Churned customers have a much smaller revolving balance which, because they don't fully pay off their credit card balance, may signify that they have less disposable income than staying customers that know they can pay off their revolving balance
- Churned customers will have a lower amount of transactions, which makes sense as they're less involved with this company and will have a smaller transaction change over time as displayed

![image](https://user-images.githubusercontent.com/64974149/135511682-9abab7e1-c73a-4e5f-81a1-67b9d238582b.png)

**A lower Total Transaction change,revolving balance, and higher Number of contacts within the past year are most correlated with a churning customer**

The following features are the most correlated (> 0.75%)
  - The months of being a customer with the bank(months on the book) and the Age are positive
  - Credit Limit and Average Open To Buy Credit Line are also positive

The following are moderately correlated (30-75%)
  - Total Transaction count and Total Relationship count are negative
  - Credit Limit and Average Utilization Ratio are negative Total Revolving balance and Average Utilization Ratio are positive Average Open To Buy and Average Utilization Ration are negative



