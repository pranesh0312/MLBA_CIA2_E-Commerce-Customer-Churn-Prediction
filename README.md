# Customer Churn Prediction Using Machine Learning

## 1. Project Overview
This project focuses on predicting customer churn in an e-commerce business using Machine Learning. It analyses customer information and purchasing behaviour to identify customers who are likely to stop purchasing from the platform.

The main objective is to help businesses identify high-risk customers, improve customer retention and make data-driven decisions.

## 2. Business Problem
E-commerce businesses face difficulties identifying customers who are likely to leave their platforms. Identifying churn only after customers become inactive can result in revenue loss and reduced customer loyalty.

This project develops a machine learning model to predict customer churn and support businesses in taking preventive retention actions.

## 3. Dataset Description
- Dataset: E-Commerce Customer Churn Dataset
- File: E Commerce Dataset.xlsx
- Total Records: 5,630
- Original Columns: 20
- Target Variable: Churn
- Input Features: 18

Target variable:
- 0 – Not Churned
- 1 – Churned

The dataset contains customer details such as tenure, satisfaction score, complaints, cashback amount, payment mode and purchasing behaviour.

## 4. Data Preprocessing
The following preprocessing techniques were performed:
- Removed the CustomerID column.
- Checked and handled missing values.
- Checked duplicate records.
- Standardised inconsistent categorical values.
- Detected outliers using the IQR method.
- Applied median imputation for numerical features.
- Applied most-frequent imputation for categorical features.
- Used StandardScaler for numerical feature scaling.
- Used OneHotEncoder for categorical feature encoding.

## 5. Machine Learning Methodology
The dataset was divided into training and testing sets using an 80:20 ratio.

Three classification algorithms were implemented:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

Random Forest was selected for the final simplified model using 10 selected customer features.

## 6. Model Performance

### Initial Model Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 79.2% | 44.0% | 84.7% | 57.9% |
| Decision Tree | 83.8% | 51.3% | 83.7% | 63.6% |
| Random Forest | 96.6% | 91.8% | 87.9% | 89.8% |

### Final Simplified Random Forest Model

| Metric | Result |
|---|---:|
| Accuracy | 95.4% |
| Precision | 84.5% |
| Recall | 88.9% |
| F1-score | 86.7% |

## 7. Business Insights
- Customers with shorter tenure may have a higher churn risk.
- Customer complaints show a positive relationship with churn.
- Customer satisfaction is an important factor to monitor.
- Customer purchasing behaviour can help identify potential churn.
- Targeted offers and customer engagement can support retention strategies.

## 8. Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook

## 9. Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── Customer_Churn_Analysis.ipynb
│
├── model/
│   └── model.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 10. How to Run the Application

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the project folder

```bash
cd Customer-Churn-Prediction
```

### Step 3: Install the required libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit application

```bash
streamlit run app.py
```

### Step 5: Open the application

Open the local URL provided by Streamlit, usually:

http://localhost:8501

## 11. Model Saving
The trained machine learning model is saved using Joblib as `model.pkl` and can be loaded for making predictions without retraining.

## 12. Limitations
- The model is based on one e-commerce dataset.
- The dataset has an imbalance between churned and non-churned customers.
- Performance may vary on new customer data.
- Further validation is required before production use.

## 13. Future Enhancements
- Deploy the application online using Streamlit Community Cloud.
- Improve performance through hyperparameter tuning.
- Add customer risk categories.
- Develop personalised customer retention recommendations.
- Monitor model performance using new customer data.

## 14. Project Links
- **GitHub Repository:** Add your repository URL.
- **Jupyter Notebook:** Add your notebook URL.
- **Live Application:** Add your Streamlit deployment URL after deployment.

---

**Project Title:** Customer Churn Prediction Using Machine Learning  
**Domain:** E-Commerce / Business Analytics  
**Objective:** Predict customer churn and support customer retention decisions.
