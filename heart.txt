"""
Heart Disease Risk Prediction Web Application with Machine Learning Workflow,
 Interactive Dashboard, and Chatbot using OpenAI Application Programming Interface (API). 

 Instructions

1. Install required libraries:
   pip install pandas numpy seaborn matplotlib scikit-learn openpyxl

2. Make sure the dataset (heart_data 2.xlsx) is in the same folder as this script.
   - If using CSV instead, update: pd.read_csv("heart_data 2.csv")

3. Run the script:
   python heart.py

This script:
- Cleans and preprocesses the dataset
- Performs EDA with plots (countplots, heatmaps, histograms)
- Trains multiple models (Logistic Regression, Decision Tree, Random Forest, etc.)
- Evaluates results with metrics, ROC curves, and confusion matrices
"""


# Importing necessary libraries

import pandas as pd            
import numpy as np             
import seaborn as sns          
import matplotlib.pyplot as plt 

# Loading and inspecting dataset

dataframe = pd.read_excel("heart_data.xlsx")  
print(dataframe.head())   # It loads the dataset into a DataFrame and shows the first few rows

dataframe.head(5)     
dataframe.tail(5)     
dataframe.shape       # It checks the number of rows and columns in dataset
dataframe.describe()  # It gives a statistical summary of the data (mean, std, min, max, etc.)

# Checking how many values are 0 in each column (to detect missing/improper values)
(dataframe == 0.0).sum()

# Checking the datatypes of each column (important for preprocessing)
print(dataframe.dtypes)

# Checking how many patients have heart disease vs no heart disease
#  'target' column: 1 = disease, 0 = no disease
print(dataframe['target'].value_counts())


# Exploratory Data Analysis

# Bar plot comparing number of males and females with and without disease
sns.countplot(x='sex', hue='target', data=dataframe, palette='coolwarm')
plt.title("Sex vs Heart Disease")
plt.xticks([0,1], ['Female','Male'])  # rename 0 -> Female, 1 -> Male
plt.show()

# Correlation heatmap which shows how features are related to each other and to target
plt.figure(figsize=(10,8))
sns.heatmap(dataframe.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()

# Histograms of all numerical features which show the distribution of values
dataframe.hist(bins=50, figsize=(20,15))

# Handle Missing or Wrong Values
(dataframe == 0.0).sum()

# Calculating median cholesterol and ignoring 0 values since they are invalid
median_cholesterol = dataframe[dataframe['cholesterol'] > 0]['cholesterol'].median()
print(median_cholesterol)

# Replacing cholesterol 0 values with median because they are invalid
dataframe['cholesterol'] = dataframe['cholesterol'].replace(0, median_cholesterol)

# Verifying  if cholesterol column still has 0 values
(dataframe == 0.0).sum()

# Plot heatmap after fixing cholesterol column
plt.figure(figsize=(10,8))
sns.heatmap(dataframe.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features (After Cleaning)")
plt.show()

# Feature Selection
# Dropping columns that are less useful or having negative correlation for heart disease risk
# Removing: 'resting bp s', 'cholesterol', 'fasting blood sugar', 'resting ecg', 'max heart rate'
dataframe1 = dataframe.drop(['resting bp s', 'cholesterol',
               'fasting blood sugar', 'resting ecg', 'max heart rate'], axis=1)

dataframe1   # Display new dataset

# Plot heatmap for reduced feature set
plt.figure(figsize=(10,8))
sns.heatmap(dataframe1.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Reduced Features")
plt.show()

for col in dataframe1.columns:
    print(f"\nColumn: {col}")
    print(dataframe1[col].value_counts())


# Train-Test Split

from sklearn.model_selection import train_test_split

X = dataframe1.drop('target', axis=1)   # independent variables (features)
y = dataframe1['target']                # dependent variable (label)

# Split dataset into train and test (80/20 split ratio), stratify keeps same ratio of target classes in proportion
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45, stratify=y
)

# Checking class distribution in train and test sets
print("Train target distribution:\n", y_train.value_counts())
print("Test target distribution:\n", y_test.value_counts())

# Comparing feature distributions in train vs test 
for col in ['sex', 'chest pain type', 'exercise angina', 'ST slope','oldpeak']:
    print(f"\n{col} - Train:")
    print(X_train[col].value_counts(normalize=True))
    print(f"{col} - Test:")
    print(X_test[col].value_counts(normalize=True))


#Data Cleaning - Fix 'oldpeak'
# Oldpeak values should not be negative, so we replace negatives with 0
X['oldpeak'] = X['oldpeak'].apply(lambda x: 0 if x < 0 else x)
X_train['oldpeak'] = X_train['oldpeak'].apply(lambda x: 0 if x < 0 else x)
X_test['oldpeak']  = X_test['oldpeak'].apply(lambda x: 0 if x < 0 else x)

print(dataframe1['oldpeak'])

# Verify oldpeak distribution after cleaning
for col in ['oldpeak']:
    print(f"\n{col} - Train:")
    print(X_train[col].value_counts(normalize=True))
    print(f"{col} - Test:")
    print(X_test[col].value_counts(normalize=True))

# Visualizing Train vs Test Distributions

dataframe1_train = X_train.copy()
dataframe1_train['target'] = y_train
dataframe1_test = X_test.copy()
dataframe1_test['target'] = y_test

cols = ['sex', 'chest pain type', 'exercise angina', 'ST slope']

# Plot side-by-side distribution of features in train vs test
for col in cols:
    plt.figure(figsize=(8,4))
    train_counts = dataframe1_train[col].value_counts(normalize=True).sort_index()
    test_counts  = dataframe1_test[col].value_counts(normalize=True).sort_index()
    df_plot = pd.DataFrame({'Train': train_counts, 'Test': test_counts})
    df_plot.plot(kind='bar', figsize=(8,4))
    plt.title(f'Train vs Test Distribution: {col}')
    plt.ylabel('Proportion')
    plt.xlabel(col)
    plt.show()


# Train Multiple Models
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Logistic Regression (baseline model)
pipeline = Pipeline([
    ('scaler', StandardScaler()),      
    ('model', LogisticRegression())    
])
pipeline.fit(X_train, y_train)             # train model
y_pred = pipeline.predict(X_test)          # predictions
print("Accuracy:", accuracy_score(y_test, y_pred))

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
pipe_dt = Pipeline([
    ('scaler', StandardScaler()),          
    ('dt', DecisionTreeClassifier(random_state=45))
])
pipe_dt.fit(X_train[cols], y_train)        
y_pred_dt = pipe_dt.predict(X_test[cols])  
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Random Forest Classifier (ensemble of decision trees)
from sklearn.ensemble import RandomForestClassifier
pipe_rf = Pipeline([
    ('scaler', StandardScaler()),  
    ('rf', RandomForestClassifier(n_estimators=100, random_state=45))
])
pipe_rf.fit(X_train, y_train)
y_pred_rf = pipe_rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Linear Regression (not ideal for classification, but tested for comparison)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
pipe_lr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])
pipe_lr_model.fit(X_train, y_train)
y_pred_lr = pipe_lr_model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred_lr))   # regression error
print("R2 Score:", r2_score(y_test, y_pred_lr))        # variance explained
# Convert regression output into classification (>=0.5 → 1, else 0)
y_pred_lr_bin = (y_pred_lr >= 0.5).astype(int)
print("Linear Regression Accuracy (binary):", accuracy_score(y_test, y_pred_lr_bin))

# K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier
pipe_knn = Pipeline([
    ('scaler', StandardScaler()),  
    ('knn', KNeighborsClassifier(n_neighbors=5))  # default k=5
])
pipe_knn.fit(X_train, y_train)
y_pred_knn = pipe_knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# Gradient Boosting (advanced boosting model)
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3, random_state=45
)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))

# Comparing Models with Metrics

from sklearn.metrics import precision_score, recall_score, f1_score

# Storing predictions from all the abovemodels
models_preds = {
    'Logistic Regression': y_pred,
    'Decision Tree': y_pred_dt,
    'Random Forest': y_pred_rf,
    'Linear Regression (binary)': y_pred_lr_bin,
    'K-Nearest Neighbors': y_pred_knn,
    'Gradient Boosting': y_pred_gb
}

metrics_list = []
# Calculating metrics for each model
for model_name, preds in models_preds.items():
    metrics_list.append({
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_test, preds), 3),
        'Precision': round(precision_score(y_test, preds), 3),
        'Recall': round(recall_score(y_test, preds), 3),
        'F1-score': round(f1_score(y_test, preds), 3),
    })

metrics_df = pd.DataFrame(metrics_list)
metrics_df = metrics_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
print(metrics_df)


# Accuracy Comparison Plot in bargraph

accuracy_dict = {name: accuracy_score(y_test, preds) for name, preds in models_preds.items()}
accuracy_table = pd.DataFrame(list(accuracy_dict.items()), columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
print(accuracy_table)

# Bar plot of accuracy scores
plt.figure(figsize=(8,5))
sns.barplot(x='Accuracy', y='Model', data=accuracy_table, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.xlim(0,1)
plt.show()


#  ROC Curves for models with different colours
from sklearn.metrics import roc_curve, auc
models_proba = {
    'Logistic Regression': pipeline,
    'Random Forest': pipe_rf,
    'Gradient Boosting': gb_model,
    'K-Nearest Neighbors': pipe_knn
}
plt.figure(figsize=(8,6))
for name, model in models_proba.items():
    try:
        y_proba = model.predict_proba(X_test)[:,1]   # probability of positive class
        fpr, tpr, _ = roc_curve(y_test, y_proba)     # false positive rate, true positive rate
        roc_auc = auc(fpr, tpr)                      # area under curve (AUC)
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')
    except:
        print(f'{name} skipped (no predict_proba)')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)
plt.show()

# Confusion Matrices
from sklearn.metrics import confusion_matrix

# Function to plot confusion matrix with labels
def plot_conf_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Generating confusion matrix for each model
plot_conf_matrix(y_test, y_pred, "Logistic Regression")
plot_conf_matrix(y_test, y_pred_dt, "Decision Tree")
plot_conf_matrix(y_test, y_pred_rf, "Random Forest")
plot_conf_matrix(y_test, y_pred_lr_bin, "Linear Regression (Binary)")
plot_conf_matrix(y_test, y_pred_knn, "KNN")
plot_conf_matrix(y_test, y_pred_gb, "Gradient Boosting")
