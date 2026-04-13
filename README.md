# 📊 Customer Churn Prediction - Week 10

## 📌 Project Overview

This project focuses on predicting customer churn using advanced **data preprocessing and feature engineering techniques**.

The goal is to identify customers who are likely to leave and help businesses take preventive actions.

---

## 🎯 Objectives

* Handle categorical data using multiple encoding techniques
* Apply feature scaling methods
* Detect and remove outliers
* Create new meaningful features
* Build a complete ML pipeline

---

## 📂 Project Structure

```

week10/

│── churn\_prediction\_pipeline.py

│── customer\_churn.csv

│── README.md

│── requirements.txt

```

---

## ⚙️ Setup Instructions

### Install Dependencies

```

pip install pandas numpy scikit-learn

```

### Run Project

```

python churn\_prediction\_pipeline.py

```

---

## 🧠 Techniques Used

### 🔹 Encoding Methods

* Label Encoding
* Binary Encoding
* One-Hot Encoding

### 🔹 Feature Engineering

* Average Charge per tenure
* Tenure groups
* Charge efficiency
* High spender indicator
* Long-term customer flag

### 🔹 Scaling Techniques

* StandardScaler
* MinMaxScaler

### 🔹 Outlier Handling

* IQR Method

---

## 🤖 Model Used

* Logistic Regression

---

## 📊 Model Performance

* Accuracy: ~80% (depends on dataset)
* Good classification balance

---

## 🔍 Key Insights

* Customers with high monthly charges tend to churn more
* Long-term customers are less likely to leave
* Contract type significantly impacts churn

---

## 🧪 Testing Evidence

* Pipeline runs end-to-end without errors
* Data successfully encoded, scaled, and cleaned
* Model predictions generated correctly

---

## 🚀 Future Improvements

* Use Random Forest / XGBoost
* Hyperparameter tuning
* Deploy model as web app

---

## 👨‍💻 Author

Rahul K N

---

## ⭐ Conclusion

This project demonstrates a complete preprocessing pipeline and is a strong addition to any data science portfolio.
