# Uber Trip Demand Analysis & Prediction

## 📌 Project Overview
This project analyzes New York City Uber pickup data to identify temporal demand patterns and build robust predictive machine learning models. By leveraging historical trip data, the objective is to forecast future ride demand, which is a critical component for optimizing fleet management and driver allocation in the ride-sharing industry.

## 📊 Dataset
The data is sourced from the NYC Taxi & Limousine Commission (TLC) FOIL response. 
* **Primary File:** `Uber-Jan-Feb-FOIL.csv`
* **Features:** Dispatching base number, Date, Active vehicles, and Trips.

## 🛠️ Tech Stack & Tools
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / VS Code

## 🧠 Methodology Workflow

### 1. Data Preprocessing
* Mapped raw TLC dispatching base codes (e.g., `B02512`, `B02764`) to their actual company names (Unter, Grun, Hinter, etc.) for better interpretability.
* Converted date strings to datetime objects for time-series manipulation.

### 2. Feature Engineering
Engineered time-based features to capture the seasonality and trends of ride-sharing demand:
* Extracted **Hour of day**, **Day of week**, and **Month**.
* Created dummy variables (one-hot encoding) for the Dispatch Bases.

### 3. Model Training & Evaluation
Evaluated multiple regression models utilizing `TimeSeriesSplit` to respect the temporal order of the data and prevent data leakage. The primary evaluation metric used was **MAPE (Mean Absolute Percentage Error)**.

| Model | MAPE Performance |
| :--- | :--- |
| **XGBoost Regressor** | **8.37%** |
| **Weighted Ensemble** | 8.60% |
| **Random Forest** | 9.61% |
| **Gradient Boosting** | 10.02% |

### 4. Ensemble Strategy
Implemented a custom Weighted Ensemble Model combining the predictions of XGBoost, Random Forest, and Gradient Boosting. Weights were assigned based on the reciprocal of each model's error rate, ensuring the most accurate model had the highest influence.

## 🚀 Key Insights
* **Top Performer:** The XGBoost model achieved the highest accuracy with a MAPE of 8.37%, making it highly effective for precise driver positioning.
* **Temporal Patterns:** Significant demand spikes occur during specific days of the week and times, highlighting the importance of time-based feature engineering in the model.

## 📁 Repository Structure
* `Notebook.ipynb`: The main Jupyter Notebook containing all data cleaning, EDA, feature engineering, and model training code.
* `Uber-Jan-Feb-FOIL.csv`: The dataset used for training and testing.
* `Uber Trip Analysis Machine Learning Project (Data Analyst).pdf`: Detailed project report and findings.

