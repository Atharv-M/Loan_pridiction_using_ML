# Model_Comparision

Data Set Link=https://drive.google.com/file/d/124cCTyCcHGkmty6eqv-bNMy7d4cjCHYL/view?usp=drive_link
presentation Video Linl=





 Loan Default Prediction and Model Evaluation Dashboard

This project aims to develop a machine learning pipeline that predicts loan default outcomes using various supervised learning algorithms. The solution is built using Python, leveraging popular libraries like pandas, NumPy, scikit-learn, Seaborn, and Matplotlib. The goal is to not only achieve accurate predictions but also compare multiple models on performance metrics and visualize key insights like feature importance, ROC curves, and training times.



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

### Why This Matters:
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


How to Run the Code

1. Place the dataset `loan_model_ready.csv` in the working directory.
2. Install required libraries using pip:



3. Run the notebook or script.





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

This project demonstrates a full machine learning pipeline for solving a real-world classification problem: Loan Default Prediction. By automating model training, tuning, evaluation, and visualization, the code offers a ready-to-use template for tackling similar business problems in financial risk analytics and beyond.


