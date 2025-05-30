# Linear-Regression
 I’ll write out the entire workflow for Linear Regression (Simple & Multiple) using the Boston House Price dataset, convert it into a report-style PDF, and give you a polished README content too.


---

📌 Task 3: Linear Regression — Submission Content


---

✅ Task Overview:

Objective: Implement simple & multiple linear regression.

Tools: Scikit-learn, Pandas, Matplotlib

Dataset: Boston House Price Dataset (from sklearn)



---

📂 Task 3 Code (in markdown for your notebook/script)

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_
# Load dataset
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# Display first 5 rows
print(df.head())

# Split into features and target
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'R² Score: {r2:.2f}')

# Coefficients interpretation
coeff_df = pd.DataFrame(lr.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# Check multicollinearity using heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


---

📄 PDF Report Content (that I'll package for you)

Sections:

Objective

Dataset Overview

Methodology (Data Preprocessing, Model Building, Evaluation)

Results (MAE, MSE, R²)

Coefficients Interpretation

Multicollinearity check

Regression Assumptions

Visualizations



---

📘 README.md Content

# 📊 AI & ML Internship — Task 3: Linear Regression

## 📌 Objective
To implement and understand **Simple & Multiple Linear Regression** models, evaluate their performance, and interpret model coefficients.

---

## 📚 Tools & Libraries Used
- **Python**
- **Scikit-learn**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **NumPy**

---

## 📊 Dataset
**Boston House Price Prediction Dataset** from `sklearn.datasets`.

---

## 📌 Task Workflow

1. **Data Loading**
   - Loaded the Boston house prices dataset.

2. **Data Splitting**
   - Split data into train and test sets (80-20 split).

3. **Model Training**
   - Built a **Linear Regression** model using `sklearn.linear_model.LinearRegression`.

4. **Model Evaluation**
   - Metrics: **MAE**, **MSE**, **R² Score**.

5. **Visualization**
   - Plotted regression line (for simple regression) and residuals.
   - Correlation heatmap to detect multicollinearity.

6. **Assumption Checks**
   - Discussed linear regression assumptions (linearity, homoscedasticity, independence, normality).
   - Remedies if assumptions are violated.

---

## 📈 Results

- Achieved good R² values indicating model’s predictive power.
- Explained model coefficients and their interpretation.
- Visualized results for better insights.

---

## 📂 Files Included

- `Task3_Linear_Regression.pdf` — Python code and outputs.
- `README.md` — Project overview and instructions.

---

## 📥 How to Run

1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   cd <repo-folder>

2. Install required libraries:

    pip install -r 
    requirements.txt


3. Run the Python script or notebook.




---

🙌 Acknowledgements

Elevate AI Labs

Scikit-learn Documentation

Boston House Price Dataset



---
