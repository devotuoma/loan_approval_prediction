Loan Approval Predictor - Complete Implementation with Detailed Document

Project Description

This project develops a machine learning system to predict loan approval decisions based on applicant information. The system automates the loan screening process by analyzing patterns in historical loan data and building predictive models that can assist financial institutions in making consistent, data-driven lending decisions.

Table of Contents
1. Project Setup and Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Model Development
5. Model Evaluation
6. Additional Analysis
6. Conclusions and Recommendations
7. Project Summary Report







Project Setup and Data Loading

Description

This stage involves setting up the development environment, importing necessary libraries, and loading the loan dataset. We establish the foundation for our machine learning pipeline by importing essential Python libraries for data manipulation, visualization, and machine learning.

Dataset

The dataset (Loan_Train.csv) contains 614 samples with 13 features including:

. Loan_ID: Unique loan ID

. Gender: Applicant's gender

. Married: Marital status

. Dependents: Number of dependents

. Education: Education level

. Self_Employed: Self-employment status

. ApplicantIncome: Applicant's income

. CoapplicantIncome: Co-applicant's income

. LoanAmount: Loan amount in thousands

. Loan_Amount_Term: Term of loan in months

. Credit_History: Credit history meets guidelines

. Property_Area: Urban/Semi-Urban/Rural

. Loan_Status: Loan approved (Y/N) - Target variable


















2. Exploratory Data Analysis (EDA)



Description

EDA is crucial for understanding the patterns, relationships, and characteristics in our loan dataset. This stage involves comprehensive data exploration including target variable distribution, feature relationships, missing value analysis, and identifying potential insights that will guide our modeling approach.


Stage Explanation Purpose: Gain deep insights into data patterns, relationships, and characteristics that will inform our modeling strategy.

Key Activities:

Target Variable Analysis: Understanding loan approval distribution (68% approved, 32% rejected) Categorical Analysis: Examining how gender, marital status, education, etc. relate to loan approval Numerical Analysis: Studying income patterns, loan amounts, and their distributions Correlation Analysis: Identifying relationships between numerical variables Advanced Insights: Creating derived features and business-relevant metrics.
<img width="1065" height="439" alt="E1" src="https://github.com/user-attachments/assets/3a5b4797-4732-43c8-a967-59bd530aa564" />

<img width="1080" height="500" alt="E2" src="https://github.com/user-attachments/assets/d508576e-374b-44f5-93a1-09b7830dc6d5" />


<img width="1075" height="496" alt="E3" src="https://github.com/user-attachments/assets/fcb12294-8615-40e9-a186-950e8f8f5da6" />
<img width="634" height="534" alt="E4" src="https://github.com/user-attachments/assets/91c10d60-291e-4a22-b060-a3b09797f4ed" />
<img width="737" height="550" alt="E5" src="https://github.com/user-attachments/assets/da7c53ff-8489-47f4-afa2-24edf9a503cb" />








Findings:

. Credit history is the strongest predictor (96% vs 8% approval rate)

. Property area shows significant variation (Semiurban: 68%, Urban: 67%, Rural: 61%)

. Higher total income correlates with better approval chances

. Moderate class imbalance exists but is manageable

Expected Outcomes:

. Clear understanding of data quality and missing values

. Identification of most influential features

. Business insights for decision-making

. Data-driven preprocessing strategy











3. Data Preprocessing

Description

Data preprocessing is critical for model performance. This stage involves cleaning the data, handling missing values, encoding categorical variables, creating new features, and preparing the dataset for machine learning algorithms. We ensure data quality and consistency while maximizing information retention.


Stage Explanation

Purpose: Transform raw data into a clean, high-quality dataset suitable for machine learning algorithms.

Key Activities:

. Missing Value Imputation: Strategic filling using mode for categorical and median for numerical variables.

. Feature Engineering: Creating 6 new derived features (Total_Income, Income_to_Loan_Ratio, etc.)

. Categorical Encoding: Converting text categories to numerical format using Label Encoding

. Outlier Detection: Identifying extreme values using IQR method (keeping them for now)

. Data Validation: Ensuring no infinite values, proper data types, and complete cases


New Features Created:

1.Total_Income: Combined household income

2.Income_to_Loan_Ratio: Debt-to-income indicator

3.Has_Coapplicant: Binary flag for joint applications

4.Income_Category: Low/Medium/High/Very High income segments

5.Loan_Category: Small/Medium/Large/Very Large loan amounts

6.Term_Category: Short/Medium/Long loan durations


Key Strategies:

1.Mode imputation for categorical missing values (most common category)

2.Median imputation for numerical missing values (robust to outliers)

3.Label encoding instead of one-hot encoding (preserves memory and works well with tree models)

4.Business-driven feature engineering based on domain knowledge


Expected Outcomes:

. Zero missing values in final dataset

. 14 features ready for modeling (8 original + 6 engineered)

. Proper data types and no infinite values

. Enhanced predictive power through derived features









4. Model Development

Description

This stage focuses on building and training two different machine learning models as required: Logistic Regression and Decision Tree. We implement proper train-test splitting, feature scaling where needed, model training with appropriate hyperparameters, and cross-validation for robust performance estimation.


Stage Explanation

Purpose: Build and train two complementary machine learning models to predict loan approval with proper validation and interpretability analysis.

Key Activities:

Train-Test Split: 80/20 split with stratification to maintain class balance

Feature Scaling: StandardScaler for Logistic Regression (tree models don't need scaling)

Model Configuration: Optimized hyperparameters to prevent overfitting

Training Process: Fit both models with timing and convergence monitoring

Cross-Validation: 5-fold CV with multiple metrics for robust evaluation

Interpretability: Analyze coefficients and feature importance


Model Configurations:

Logistic Regression
L2 regularization (C=1.0) to prevent overfitting liblinear solver for small datasets StandardScaler preprocessing Maximum 1000 iterations for convergence

Decision Tree
Max depth 8 to prevent overfitting Min 20 samples per split,








5. Model performance is evaluated using:

. Accuracy

. Precision

. Recall

. F1-score

. ROC-AUC score

. Confusion matrix
<img width="532" height="550" alt="L1" src="https://github.com/user-attachments/assets/ec3c4618-678d-42a2-b7f3-faa05c4697a5" />
<img width="565" height="560" alt="L2" src="https://github.com/user-attachments/assets/f1f82237-102f-4938-8e42-e48d2faaa10e" />
<img width="796" height="486" alt="L3" src="https://github.com/user-attachments/assets/c099e7fc-609a-453d-84b5-d3602b2ceeff" />
<img width="791" height="447" alt="L4" src="https://github.com/user-attachments/assets/a4e45f61-2675-4e2c-8dbb-cac17d9c051b" />











6. Additional Analysis and Recommendations

   FINAL RECOMMENDATIONS
   
CREDIT HISTORY is the most important factor - Prioritise applicants with good credit history

INCOME VERIFICATION should be thorough - Total income is a strong predictor
   
PROPERTY AREA matters - Urban applications show different approval patterns
   
Consider implementing a HYBRID APPROACH using both models for different scenarios

REGULAR MODEL RETRAINING is recommended as new data becomes available
   
Implement EXPLAINABLE AI features to help loan officers understand decisions
    
Consider additional features like employment history and existing loans
    
Set up MONITORING SYSTEMS to track model performance over time





7. Requirements


To run this notebook, you'll need:

Python 3.x

Jupyter Notebook

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn







8.  Usage

i. Clone the repository

ii. Install required packages: pip install -r requirements.txt

iii. Open the Jupyter notebook: jupyter notebook Loan_Approval_Prediction.ipynb

iv. Run all cells to reproduce the analysis






9. Future Improvements

i. Handle class imbalance using techniques like SMOTE

ii. Experiment with more advanced models (Random Forest, XGBoost)

iii. Perform hyperparameter tuning

iv. Deploy the model as a web application





10. Contributors


Otuoma Erick - Team Lead

Susan Muthoki

Martsellah Osachi

Belinder Adhiambo

Sharon Ngala

Imani Lunjala










