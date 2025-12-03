-----

# 🧪 Data Science & Machine Learning Internship Portfolio

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![MLOps](https://img.shields.io/badge/MLOps-Docker%20%7C%20FastAPI-green)

## 📌 Overview
This repository contains the completed tasks for my **Data Science & Machine Learning Internship**. The project journey covers the entire data pipeline: from raw data analysis and cleaning to building predictive models, creating interactive dashboards, automating pipelines, and finally deploying the solution using MLOps best practices.

## 📂 Repository Structure


├── 📁 Task 1 - EDA & Data Cleaning
  
├── 📁 Task 2 - Sales Forecasting (Regression)

├── 📁 Task 3 - Customer Churn (Classification)

├── 📁 Task 4 - Forecasting Dashboard

├── 📁 Task 5 - Automated Pipeline

├── 📁 Task 6 - MLOps & Deployment

└── README.md


-----

## 🛠️ Project Details

### 📊 Task 1: Exploratory Data Analysis (EDA) & Cleaning

**Focus:** Mastering the foundations of the data science workflow.

  * **Goal:** Analyze `globalmart_sales.csv` to understand sales trends, correlations, and data quality.
  * **Key Actions:**
      * cleaned missing values and handled duplicates.
      * Performed outlier detection.
      * Visualized top-selling products and customer demographics using **Matplotlib** and **Seaborn**.
  * **Outcome:** A comprehensive insight report on GlobalMart's sales trends.

### 📈 Task 2: Predicting Future Sales (Regression)

**Focus:** Time-series forecasting and inventory planning.

  * **Goal:** Build regression models to predict future product sales.
  * **Key Actions:**
      * Feature Engineering: Created time-based features (month, day, promo flags).
      * Model Training: Compared **Linear Regression**, **Random Forest**, and **XGBoost**.
      * Evaluation: Assessed performance using RMSE, MAE, and R² scores.
  * **Outcome:** A robust predictive model capable of forecasting inventory needs.

### 👥 Task 3: Customer Churn Prediction (Classification)

**Focus:** Identifying at-risk customers for retention campaigns.

  * **Goal:** Create a classifier to predict whether a customer will churn.
  * **Key Actions:**
      * Preprocessing: Encoded categorical variables and scaled numeric features.
      * Handling Imbalance: Used techniques to manage class imbalance.
      * Explainability: Generated **SHAP values** to interpret feature importance and justify model decisions.
  * **Outcome:** A production-ready classification notebook with detailed evaluation metrics (F1-score, ROC-AUC).

### 🖥️ Task 4: Demand Forecasting Dashboard

**Focus:** Visualizing insights for stakeholders.

  * **Goal:** Build an interactive dashboard using the regression model from Task 2.
  * **Key Actions:**
      * Developed a web UI using **Streamlit**.
      * Implemented time-range filters and product selectors.
      * Visualized predictions with confidence intervals and KPIs.
  * **Outcome:** An interactive tool allowing managers to view and download sales forecasts.

### 🔄 Task 5: Automated Data Pipeline

**Focus:** Automating data flows and model retraining.

  * **Goal:** Build an end-to-end pipeline that ingests new data and retrains models.
  * **Key Actions:**
      * Connected to a simulated data source.
      * Automated preprocessing and feature generation scripts.
      * Implemented model versioning using `joblib`.
      * Uses github actions to run train.py and build automatically when new dataset uploaded.
  * **Outcome:** A reproducible pipeline ensuring the model stays up-to-date without manual intervention.

### 🚀 Task 6: MLOps — Deployment & API Serving (Capstone)

**Focus:** Bridging the gap between experimentation and production.

  * **Goal:** Deploy the model as a high-performance API.
  * **Key Actions:**
      * **API Development:** Built a REST API using **FastAPI** with Pydantic for input validation.
      * **Containerization:** Packaged the application using **Docker** to ensure consistency across environments.
      * **Cloud Deployment:** Deployed the Docker container to a cloud provider (Render/Hugging Face Spaces).
  * **Outcome:** A live public URL where users can send JSON requests and receive real-time machine learning predictions.

-----

## 🧰 Tech Stack

  * **Languages:** Python
  * **Data Manipulation:** Pandas, NumPy
  * **Visualization:** Matplotlib, Seaborn, Plotly
  * **Machine Learning:** Scikit-Learn, XGBoost, SHAP
  * **Web Frameworks:** Streamlit, FastAPI
  * **DevOps/MLOps:** Docker, Uvicorn, Render/HuggingFace
  * **Tools Used:** Google Colab, VS Code, Git & GitHub, GitHub Actions, Postman, Docker Hub, Docker Desktop

-----



