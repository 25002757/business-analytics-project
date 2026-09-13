# Customer Churn Prediction
## Business Problem

Customer churn is an important business problem because losing customers can affect revenue and customer retention.
## Objective

The objective of this project is to develop a machine learning model that predicts whether a customer is likely to churn.

## Dataset

The dataset contains 10,205 customer records and 16 variables related to customer demographics, usage behaviour, satisfaction, complaints, orders, and churn.

## Methodology

The project involved data understanding, data cleaning, exploratory data analysis, preprocessing of categorical variables, and development of a machine learning classification model.

## Machine Learning Model

A Random Forest Classifier was used to predict customer churn.

## Model Performance

Accuracy: 99.90%

Precision: 99.79%

Recall: 99.79%

F1 Score: 99.79%

## Exploratory Data Analysis

The project includes analysis of:

- Churn distribution
- Customer churn by satisfaction score
- Customer churn by complaint status
- Customer tenure by churn status
- Days since last order by churn status

## Streamlit Application

The Streamlit application presents the business problem, data insights, and customer churn prediction interface.

## Project Files

- `app.py` – Streamlit application
- `model.pkl` – trained machine learning model
- `ecommerce_churn_data.csv` – dataset
- `model_development.ipynb` – model development notebook
- `requirements.txt` – required Python libraries

## How to Run

Install the required libraries using:

`pip install -r requirements.txt`

Run the Streamlit application using:

`streamlit run app.py`

## Deployment

Deployed Streamlit application link: To be added

GitHub repository: This repository
