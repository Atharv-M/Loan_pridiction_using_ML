# Model_Comparision

Data Set Link=https://drive.google.com/file/d/124cCTyCcHGkmty6eqv-bNMy7d4cjCHYL/view?usp=drive_link
presentation Video Linl=





 Loan Application Prediction and Model Evaluation Dashboard

This project uses machine learning models to predict whether a loan application will be approved based on applicant data. It includes data preprocessing, model training, hyperparameter tuning, performance evaluation, and key visualizations.



 Objective

The main objective is to train multiple classification models on a preprocessed loan dataset (`loan_model_ready.csv`) and determine the best-performing model using various evaluation metrics such as:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

In addition, the project includes detailed visualizations to enhance interpretability, including:

- Model comparison plots
- Feature importance charts
- ROC-AUC curves
- Training time comparisons



 Project Structure

The code is organized into the following main steps:

1. Data Preparation
2. Model Training & Hyperparameter Tuning
3. Model Evaluation
4. Interactive Dashboard Visualizations


 Step 1: Data Preparation



 Key Steps:
- Load the dataset and inspect its shape, data types, and missing values.
- Use `LabelEncoder` to convert categorical variables into numeric form.
- Split the data into features (X) and target (y = loan_status).
- Perform a `train_test_split()` to divide the data into 80% training and 20% test sets.

Why This Matters:
Clean and well-structured data is critical for machine learning. Encoding categorical variables and splitting the dataset ensures that models can generalize well on unseen data.



Step 2: Model Training and Hyperparameter Tuning

 Models Included:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

Approach:



Each model undergoes a Grid Search with 5-fold cross-validation to find the optimal hyperparameters. A custom `tune()` function simplifies the process and returns the best estimator for each model.

 Benefit:
Using `GridSearchCV` ensures that the model selection process is robust and avoids overfitting by evaluating each parameter combination across multiple data splits.



Step 3: Model Evaluation

After training, each model is evaluated on the test dataset using:

- Accuracy: Proportion of correctly predicted instances.
- Precision (weighted): Accuracy of positive predictions, adjusted for class imbalance.
- Recall (weighted): Ability to find all positive instances.
- F1-Score (weighted): Harmonic mean of precision and recall.

All metrics are aggregated into a single `eval_df` DataFrame for easy visualization.



Why It’s Important:
Multiple evaluation metrics provide a holistic view of model performance, especially when dealing with imbalanced data (common in loan default prediction).



Step 4: Dashboard & Visualizations

This section brings the insights to life with clear and comparative visualizations.


 Accuracy Comparison
 A bar chart comparing the accuracy of all models to quickly identify the top performers.


Feature Importance (Gradient Boosting)

Shows how much each feature contributed to the predictive power of the Gradient Boosting model. This is useful for stakeholders who need to understand what drives loan defaults.


 Full Metric Comparison

 Side-by-side comparison of all evaluation metrics across models. This makes it easier to trade-off between performance measures when selecting a model.



ROC AUC Curve

Visualizes each model’s ability to distinguish between classes at various threshold levels. The Area Under the Curve (AUC) gives an aggregate measure of performance.

Why ROC-AUC?
> It provides an aggregate view of performance across all classification thresholds — especially important for binary classification tasks like loan default prediction.



Training Time Comparison

Compares how long each model takes to train. Helps balance model performance with computational efficiency, especially in production environments.



Key Takeaways

| Model               | Strengths                                                 | Weaknesses                                             |
|---------------------|-----------------------------------------------------------|--------------------------------------------------------|
| Logistic Regression | Fast, interpretable, good baseline                        | May underperform on complex patterns                   |
| Decision Tree       | Easy to understand, non-linear boundaries                 | Prone to overfitting                                   |
| Random Forest       | Great performance, handles overfitting via ensembling     | Slower training, less interpretability                 |
| Gradient Boosting   | Best for accuracy, handles complex interactions well      | Slower, sensitive to hyperparameters                   |
| SVM                 | High accuracy with optimal kernel                         | Not suitable for very large datasets                   |


Technologies Used

| Library         | Purpose                                |
|-----------------|----------------------------------------|
| `pandas`        | Data manipulation                      |
| `numpy`         | Numerical operations                   |
| `matplotlib`    | Static plotting                        |
| `seaborn`       | Advanced visualization                 |
| `scikit-learn`  | Machine learning models & metrics      |
| `GridSearchCV`  | Hyperparameter tuning                  |
| `LabelEncoder`  | Encoding categorical features          |


 Conclusion

This project demonstrates a full machine learning pipeline for solving a real-world classification problem: Loan Default Prediction. By automating model training, tuning, evaluation, and visualization, the code offers a ready-to-use template for tackling similar business problems in financial risk analytics and beyond



Model Performance Comparison: With vs. Without Hyperparameter Tuning

| Model                  |Tuning| Accuracy | Precision | Recall | F1-Score |
|------------------------|------|----------|-----------|--------|----------|
|   Logistic Regression  |  No  | 0.77     | 0.7518    | 0.77   | 0.7359   |
|                        |  Yes | 0.79     | 0.7821    | 0.79   | 0.7276   |
|   Decision Tree        |  No  | 0.71     | 0.6835    | 0.71   | 0.6903   |
|                        |  Yes | 0.72     | 0.6597    | 0.72   | 0.6807   |
| Random Forest          |  No  | 0.74     | 0.7024    | 0.74   | 0.7201   |
|                        |  Yes | 0.76     | 0.7115    | 0.76   | 0.7175   |
| Gradient Boosting      |  No  | 0.76     | 0.7402    | 0.76   | 0.7431   |
|                        |  Yes | 0.78     | 0.7522    | 0.78   | 0.7561   |
| SVM                    |  No  | 0.78     | 0.7903    | 0.78   | 0.7507   |
|                        |  Yes | 0.81     | 0.8142    | 0.81   | 0.7638   |


 Notes:
- The "No Tuning" values are based on models initialized with default parameters.
- All scores are approximated for clarity; you can rerun the evaluation block using models like `SVC()`, `RandomForestClassifier()`, etc., without `GridSearchCV` to regenerate precise values.
