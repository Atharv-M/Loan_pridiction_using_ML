""" Comparison of Different Classification Models """


# === Common Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# === Scikit-learn Modules ===
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)





#Step 1: Data Preparation


df = pd.read_csv("loan_model_ready.csv")


# Show first 5 rows
print("🔹 First 5 rows:")
display(df.head())

# Dataset shape (rows, columns)
print("\n🔹 Dataset shape:")
print(df.shape)

# Column data types and null values
print("\n🔹 Data types and null values:")
print(df.info())

# Summary statistics for numeric columns
print("\n🔹 Summary statistics:")
display(df.describe())

# Count of missing values
print("\n🔹 Missing values per column:")
print(df.isnull().sum())

# List of all columns
print("\n🔹 All column names:")
print(df.columns.tolist())

# Count values of target column (Update this if your target is different)
print("\n🔹 Target column value counts:")
print(df['loan_status'].value_counts()) 



# Encode categorical columns
for col in df.select_dtypes('object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





#Step 2: Model Training + Hyperparameter Tuning

def tune(name, model, params):
    grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    return name, grid.best_estimator_

models = dict([
    tune("Logistic", LogisticRegression(max_iter=1000), {
        'C': [0.1, 1], 'solver': ['liblinear']
    }),
    tune("Tree", DecisionTreeClassifier(), {
        'max_depth': [5, 10], 'criterion': ['gini', 'entropy']
    }),
    tune("Forest", RandomForestClassifier(), {
        'n_estimators': [100, 200], 'max_depth': [10]
    }),
    tune("Boosting", GradientBoostingClassifier(), {
        'n_estimators': [100], 'learning_rate': [0.05, 0.1]
    }),
])


#


# Step 3: Model Evaluation

evals = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    evals.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    })

eval_df = pd.DataFrame(evals)
print(eval_df)





#Step 4: Dashboard
#🔸 Accuracy Comparison
sns.barplot(x="Model", y="Accuracy", data=eval_df)
plt.title("Accuracy Comparison")
plt.ylim(0, 1)
plt.show()





#🔸 Feature Importance

feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': models["Boosting"].feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(data=feat_imp, y='Feature', x='Importance')
plt.title("Feature Importance (Gradient Boosting)")
plt.show()



# Melt the DataFrame to long format for seaborn
eval_melted = eval_df.melt(id_vars="Model", var_name="Metric", value_name="Score")


# Plot Model Performance Comparision


plt.figure(figsize=(12, 6))
sns.barplot(data=eval_melted, x="Model", y="Score", hue="Metric", palette="Set2")

plt.title(" Model Performance Comparison", fontsize=16)
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.legend(loc="lower right")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#ROC AUC Curve
y_test_bin = label_binarize(y_test, classes=[0, 1])
for name, model in models.items():
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test_bin, y_score)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC AUC Comparison")
plt.legend()
plt.show()





#Training Time Comparison
times = []
for name, model in models.items():
   start = time.time()
   model.fit(X_train, y_train)
   times.append({"Model": name, "Time": time.time() - start})

time_df = pd.DataFrame(times)
sns.barplot(data=time_df, x="Model", y="Time")
plt.title("Training Time Comparison")
plt.show()






