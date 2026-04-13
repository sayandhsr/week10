import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("🚀 Loading Data...")

# ================================
# LOAD DATA
# ================================
df = pd.read_csv(r"C:\Users\Dell\OneDrive\Documents\devoloper arena\week10\customer_churn (1).csv")

print("✅ Data Loaded")
print(df.head())

# ================================
# CLEAN DATA
# ================================
df.dropna(inplace=True)

# ================================
# ENCODING TARGET
# ================================
le = LabelEncoder()

if df['Churn'].dtype == 'object':
    df['Churn'] = le.fit_transform(df['Churn'])

# ================================
# HANDLE CATEGORICAL COLUMNS
# ================================

# Convert Yes/No
df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})

# Convert all remaining categorical columns
df = pd.get_dummies(df, drop_first=True)

# ================================
# FEATURE ENGINEERING
# ================================
print("\n🧠 Feature Engineering...")

df['Charge_per_Tenure'] = df['MonthlyCharges'] / (df['Tenure'] + 1)
df['Total_per_Month'] = df['TotalCharges'] / (df['Tenure'] + 1)
df['High_Spender'] = (df['MonthlyCharges'] > df['MonthlyCharges'].mean()).astype(int)
df['Long_Term'] = (df['Tenure'] > 24).astype(int)

df['Tenure_Group'] = pd.cut(
    df['Tenure'],
    bins=[0,12,24,48,60],
    labels=[0,1,2,3],
    include_lowest=True
)

# FIX NaN safely
df['Tenure_Group'] = df['Tenure_Group'].cat.codes

# ================================
# OUTLIER REMOVAL
# ================================
Q1 = df['MonthlyCharges'].quantile(0.25)
Q3 = df['MonthlyCharges'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['MonthlyCharges'] >= Q1 - 1.5*IQR) &
        (df['MonthlyCharges'] <= Q3 + 1.5*IQR)]

# ================================
# FEATURES & TARGET
# ================================
X = df.drop('Churn', axis=1)
y = df['Churn']

# ================================
# SCALING
# ================================
scaler1 = StandardScaler()
scaler2 = MinMaxScaler()

X_scaled = scaler1.fit_transform(X)
X_scaled = scaler2.fit_transform(X_scaled)

# ================================
# TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ================================
# MODEL
# ================================
print("\n🤖 Training Model...")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ================================
# EVALUATION
# ================================
y_pred = model.predict(X_test)

print("\n📊 MODEL PERFORMANCE")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\n✅ WEEK 10 COMPLETED SUCCESSFULLY")