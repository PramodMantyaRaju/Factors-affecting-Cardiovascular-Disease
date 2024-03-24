# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 17:28:24 2023

@author: Pramod
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('cardio_train.csv')

## Data Exploration & Data Preprocessing

# Displaying the first few rows
print(df.head())

# Let's look at basic statistics
print(df.describe(include='all').transpose())

# Checking if we have Null values 
print(df.isnull().sum())

# Checking if the target variable is balanced
target_variable = df['cardio'].value_counts()
print(target_variable)
    
# Plotting it on bar-graph
import matplotlib.pyplot as plt
plt.bar(target_variable.index, target_variable.values, color =['blue', 'orange'])
plt.xlabel('Cardiovascular disease')
plt.ylabel('Count')
plt.title('Count of total samples with/without cardiovascular disease')
for i, value in enumerate(target_variable.values):
    plt.text(i, value + 500, str(value), ha='center' , va='bottom')
plt.show()

# Let's add Age in Years and BMI
df['age_years'] = df['age']// 365
df ['BMI'] = df['weight'] / (df['height'] ** 2) 

# Let's remove outliers
df.drop(df[(df['height'] > df['height'].quantile(0.975)) | (df['height'] < df['height'].quantile(0.025))].index,inplace=True) #remove outliers => remove heights that fall below 2.5% or above 97.5% of a given range
df.drop(df[(df['weight'] > df['weight'].quantile(0.975)) | (df['weight'] < df['weight'].quantile(0.025))].index,inplace=True) #remove outliers => remove weights that fall below 2.5% or above 97.5% of a given range
df.drop(df[(df['ap_hi'] > df['ap_hi'].quantile(0.975)) | (df['ap_hi'] < df['ap_hi'].quantile(0.025))].index,inplace=True) #remove outliers => remove systolic blood pressure values that fall below 2.5% or above 97.5% of a given range
df.drop(df[(df['ap_lo'] > df['ap_lo'].quantile(0.975)) | (df['ap_lo'] < df['ap_lo'].quantile(0.025))].index,inplace=True) #remove outliers => remove diastolic blood pressure values that fall below 2.5% or above 97.5% of a given range
len(df)

# Let's see how many individuals between age 30 to 40 have/do not have heart disease
filtered_df = df[(df['age_years'] >= 30) & (df['age_years'] <= 40)]

heart_disease_counts = filtered_df['cardio'].value_counts() # Count the occurrences of heart disease within the filtered age group

plt.bar(heart_disease_counts.index, heart_disease_counts.values, color=['blue', 'orange']) # Plot the bar graph
plt.xlabel('Cardiovascular disease')
plt.ylabel('Count')
plt.title('Count of individuals with age between 30 and 40 with/without cardiovascular disease')


for i, value in enumerate(heart_disease_counts.values): # Add annotations
    plt.text(i, value + 50, str(value), ha='center', va='bottom')

plt.show()

# Let's see the correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = df.corr() # Compute the correlation matrix

plt.figure(figsize=(12, 10)) # Set up the matplotlib figure
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5) # Create a heatmap using seaborn
plt.title('Correlation Matrix') # Add title
plt.show() # Show the plot

# Let's see Systolic Blood Pressure Distribution by Cardiovascular Disease
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='cardio', y='ap_hi')
plt.title('Systolic Blood Pressure Distribution by Cardiovascular Disease')
plt.show()

# Let's see Diastolic Blood Pressure Distribution by Cardiovascular Disease
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='cardio', y='ap_lo')
plt.title('Diastolic Blood Pressure Distribution by Cardiovascular Disease')
plt.show()

# Let's see exposure to CVD based on age between 30 to 60
plt.figure(figsize=(11, 8))
sns.countplot(x='age_years', hue='cardio', data=df[(df['age_years'] >= 30) & (df['age_years'] <= 60)], palette="Set2")
plt.title('Exposure to Cardiovascular Disease based on Age (30 to 60)')
plt.show()

# Checking if the target variable is balanced
target_variable = df['cardio'].value_counts()
print(target_variable)
    
# Plotting it on bar-graph
import matplotlib.pyplot as plt
plt.bar(target_variable.index, target_variable.values, color =['blue', 'orange'])
plt.xlabel('Cardiovascular disease')
plt.ylabel('Count')
plt.title('Count of total samples with/without cardiovascular disease')
for i, value in enumerate(target_variable.values):
    plt.text(i, value + 500, str(value), ha='center' , va='bottom')
plt.show()


# Create dummy variables for 'cholesterol' attribute 
cholesterol_dummies = pd.get_dummies(df['cholesterol'], prefix='cholesterol')

df = pd.concat([df, cholesterol_dummies], axis=1) # Concatenate the dummy variables with the original DataFrame

df = df.drop('cholesterol', axis=1) # Drop the original "cholesterol" column

# Display the updated DataFrame
print(df.head())

## Logistic Regression

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define features (X) and target variable (y)
X = df[['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol_1', 'cholesterol_2', 'cholesterol_3', 'gluc', 'smoke', 'alco', 'active', 'BMI']]
y = df['cardio']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Using statsmodels for logistic regression
X_train_scaled_int = sm.add_constant(X_train_scaled)   # Add intercept manually
logit_model = sm.Logit(y_train, X_train_scaled_int).fit()

# Print the summary
print(logit_model.summary())

# Extract p-values
p_values = logit_model.pvalues

# Identify features with p-values greater than 0.05
insignificant_features = p_values[p_values > 0.05].index.tolist()

# Print insignificant feature names
print("Insignificant Features:")
print(insignificant_features)

# Display the updated DataFrame after removing insignificant features
print(X_train.head())

# Map the indices back to original feature names
feature_names = X.columns.tolist()
insignificant_feature_names = [feature_names[int(feature[1:]) - 1] for feature in insignificant_features if feature != 'const']

# Print insignificant feature names
print("Insignificant Feature Names:")
print(insignificant_feature_names)

# Standardize features for the final training data
X_train_scaled_final = scaler.transform(X_train)

# Refit the model using the whole training set
final_model = LogisticRegression(penalty='none', solver='newton-cg', max_iter=10000)
final_model.fit(X_train_scaled_final, y_train)

# Evaluate on the test set
X_test_scaled_final = scaler.transform(X_test)

# Make predictions on the test set
y_prob_test = final_model.predict_proba(X_test_scaled_final)[:, 1]
y_pred_test = np.where(y_prob_test > 0.5, 1, 0)

# Calculate metrics on the test set
confmat_test = confusion_matrix(y_test, y_pred_test)
accuracy_test = accuracy_score(y_test, y_pred_test)

print("\nConfusion Matrix (Test Set):")
print(confmat_test)
print("Accuracy:", accuracy_test)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob_test, pos_label=1)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Calculate AUC
auc_test = roc_auc_score(y_test, y_prob_test)
print("AUC (Test Set):", auc_test)

# Confusion Matrix Visualization
sns.heatmap(confmat_test, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Test Set)')
plt.show()

# ROC Curve Visualization 
plt.plot(fpr, tpr, label=f'AUC = {auc_test:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Feature Importance Visualization 
feature_importance = final_model.coef_[0]
feature_names = X_train.columns

plt.barh(feature_names, feature_importance)
plt.xlabel('Coefficient Magnitude')
plt.title('Logistic Regression Coefficients')
plt.show()


#KNN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# Suppress FutureWarnings from scikit-learn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Split the data into training, validation, and test sets for classification
X_train_valid_classification, X_test_classification, y_train_valid_classification, y_test_classification = train_test_split(
    X, y, test_size=0.3, random_state=123
)
X_train_classification, X_valid_classification, y_train_classification, y_valid_classification = train_test_split(
    X_train_valid_classification, y_train_valid_classification, test_size=0.30, random_state=123
)

# Standardize features for classification
scaler_classification = StandardScaler()
X_train_valid_classification_tran = pd.DataFrame(scaler_classification.fit_transform(X_train_valid_classification))
X_train_classification_tran = pd.DataFrame(scaler_classification.transform(X_train_classification))
X_valid_classification_tran = pd.DataFrame(scaler_classification.transform(X_valid_classification))
X_test_classification_tran = pd.DataFrame(scaler_classification.transform(X_test_classification))

# Run a KNN classifier to find the best K value for classification
valid_misclf = []
for k in range(1, 11):
    knn_classification = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    knn_classification.fit(X_train_classification_tran, y_train_classification)

    y_hat_classification = knn_classification.predict(X_valid_classification_tran)
    valid_misclf.append(np.mean(y_hat_classification != y_valid_classification))

# Find the best K value for classification
bestK_classification = np.argmin(valid_misclf) + 1
print("\nBest K for Classification:", bestK_classification)

# Train the final classification model with the optimal K
knn_classification_final = KNeighborsClassifier(n_neighbors=bestK_classification, weights='uniform')
knn_classification_final.fit(X_train_valid_classification_tran, y_train_valid_classification)

# Evaluate on the test set for classification
y_pred_classification = knn_classification_final.predict(X_test_classification_tran)
misclf_classification = np.mean(y_pred_classification != y_test_classification)

print("\nMisclassification Rate for Classification on Test Set:", misclf_classification)

# Plot ROC curve for classification
fpr, tpr, thresholds = roc_curve(y_test_classification, y_pred_classification, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Classification')
plt.legend(loc='lower right')
plt.show()

# Print the trained KNN classification model
print("\nKNN Classification Model:")
print(knn_classification_final)

# Simulate the impact of a 10% increase in specific features on predicted probability


# Simulate the impact of a 10% increase in specific features on predicted probability
percentage_increase = 0.10

# Cholesterol Impact
X_chol_increase = X_test_classification.copy()
X_chol_increase['cholesterol_1'] = X_chol_increase['cholesterol_1'] * (1 + percentage_increase)
X_chol_increase['cholesterol_2'] = X_chol_increase['cholesterol_2'] * (1 + percentage_increase)
X_chol_increase['cholesterol_3'] = X_chol_increase['cholesterol_3'] * (1 + percentage_increase)

# Glucose Impact
X_gluc_increase = X_test_classification.copy()
X_gluc_increase['gluc'] = X_gluc_increase['gluc'] * (1 + percentage_increase)

# Age Impact
X_age_increase = X_test_classification.copy()
# Replace 'age' with the actual column name for age in your dataset
X_age_increase['age_years'] = X_age_increase['age_years'] + (percentage_increase * X_age_increase['age_years'].mean())


# Weight Impact
X_weight_increase = X_test_classification.copy()
X_weight_increase['weight'] = X_weight_increase['weight'] * (1 + percentage_increase)

# Calculate the impact on predicted probability
cholesterol_impact_percentage = 100 * (knn_classification_final.predict_proba(X_chol_increase)[:, 1].mean() - knn_classification_final.predict_proba(X_test_classification_tran)[:, 1].mean())
glucose_impact_percentage = 100 * (knn_classification_final.predict_proba(X_gluc_increase)[:, 1].mean() - knn_classification_final.predict_proba(X_test_classification_tran)[:, 1].mean())
age_impact_percentage = 100 * (knn_classification_final.predict_proba(X_age_increase)[:, 1].mean() - knn_classification_final.predict_proba(X_test_classification_tran)[:, 1].mean())
weight_impact_percentage = 100 * (knn_classification_final.predict_proba(X_weight_increase)[:, 1].mean() - knn_classification_final.predict_proba(X_test_classification_tran)[:, 1].mean())

# Combined Effect
combined_effect_percentage = (cholesterol_impact_percentage + glucose_impact_percentage +
                              age_impact_percentage + weight_impact_percentage) / 4

print("\nImpact of a 10% Increase in Features on Predicted Probability:")
print(f"Cholesterol Impact: {cholesterol_impact_percentage:.2f}%")
print(f"Glucose Impact: {glucose_impact_percentage:.2f}%")
print(f"Age Impact: {age_impact_percentage:.2f}%")
print(f"Weight Impact: {weight_impact_percentage:.2f}%")
print(f"Combined Effect: {combined_effect_percentage:.2f}%")

###Decision Tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix

# Assuming X and y are your features and target variable
# Replace 'YourDataFrame' with the actual name of your DataFrame

# Example:
# X = YourDataFrame[['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']]
# y = YourDataFrame['cardio']

# Split the data into training, validation, and test sets for classification
X_train_valid_classification, X_test_classification, y_train_valid_classification, y_test_classification = train_test_split(
    X, y, test_size=0.25, random_state=123
)

# Hyperparameter Tuning for Classification
grid_clf = {'max_depth': np.arange(10, 31),
            'max_leaf_nodes': np.arange(20, 51)}

clf = DecisionTreeClassifier(random_state=1)
clf_grid_search = GridSearchCV(clf, grid_clf, cv=5, scoring='accuracy', n_jobs=-1)
clf_grid_search.fit(X_train_valid_classification, y_train_valid_classification)

# Report the best parameters and accuracy on the test set for Classification
print("\nBest Parameters for Classification:", clf_grid_search.best_params_)
print("Accuracy on Test Set for Classification:",
      accuracy_score(clf_grid_search.predict(X_test_classification), y_test_classification))


# Plotting Feature Importance
feature_importance = clf_grid_search.best_estimator_.feature_importances_
features = X.columns
plt.barh(features, feature_importance)
plt.xlabel('Feature Importance')
plt.title('Decision Tree Feature Importance')
plt.show()

# Print and Plot Classification Report and Confusion Matrix
print(classification_report(y_test_classification, clf_grid_search.predict(X_test_classification)))
conf_matrix = confusion_matrix(y_test_classification, clf_grid_search.predict(X_test_classification))
print("Confusion Matrix:")
print(conf_matrix)

plot_confusion_matrix(clf_grid_search.best_estimator_, X_test_classification, y_test_classification, display_labels=['No Disease', 'Disease'])
plt.show()

##SVM
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# Standardize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize SVM classifier
svm_classifier = SVC(kernel='rbf', random_state=1)

# Train the SVM classifier
svm_classifier.fit(X_train_scaled, y_train)

# Predictions on the test set
y_pred = svm_classifier.predict(X_test_scaled)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
plot_confusion_matrix(svm_classifier, X_test_scaled, y_test, display_labels=['No Disease', 'Disease'])



#Plot for SVM

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve

# Plot the decision boundary
def plot_decision_boundary(X, y, model, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', marker='o')

    # Highlight support vectors
    sv_indices = model.support_
    plt.scatter(X[sv_indices, 0], X[sv_indices, 1], c='red', marker='x', s=100, linewidths=1, label='Support Vectors')

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Plot the decision boundary for the first two features (assuming a binary classification)
X_2d = X_train_scaled[:, :2]
svm_classifier_2d = SVC(kernel='rbf', random_state=1)
svm_classifier_2d.fit(X_2d, y_train)
plot_decision_boundary(X_2d, y_train, svm_classifier_2d, 'Decision Boundary (First Two Features)')

# Plot ROC curve
plot_roc_curve(svm_classifier, X_test_scaled, y_test)
plt.title('ROC Curve for SVM')
plt.show()


## Random Forest

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# Split the data into training, validation, and test sets for classification
X_train_valid_classification, X_test_classification, y_train_valid_classification, y_test_classification = train_test_split(
    X, y, test_size=0.30, random_state=123
)
X_train_classification, X_valid_classification, y_train_classification, y_valid_classification = train_test_split(
    X_train_valid_classification, y_train_valid_classification, test_size=0.30, random_state=123
)

# Train and tune Random Forest
rf_grid = {'n_estimators': np.linspace(100, 1000, 10, dtype=int)}
RF = GridSearchCV(RandomForestClassifier(min_samples_leaf=10, random_state=10, max_features='sqrt'),
                  param_grid=rf_grid, cv=5, n_jobs=-1, scoring='f1')
RF.fit(X_train_classification, y_train_classification)

# Print the best parameters
best_params_rf = RF.best_params_
print("Best Parameters for Random Forest:", best_params_rf)

# Perform evaluation on the test set
y_pred_classification_rf = RF.predict(X_test_classification)
f1_rf = f1_score(y_test_classification, y_pred_classification_rf)
print("F1 Score for Random Forest on Test Set:", f1_rf)

# ROC Curve
fpr_rf, tpr_rf, _ = roc_curve(y_test_classification, RF.predict_proba(X_test_classification)[:, 1])

# Plot ROC Curve
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.show()

# AUC Score
auc_rf = roc_auc_score(y_test_classification, RF.predict_proba(X_test_classification)[:, 1])
print("AUC Score for Random Forest:", auc_rf)

# Confusion Matrix
conf_matrix_rf = confusion_matrix(y_true=y_test_classification, y_pred=y_pred_classification_rf)
print("Confusion Matrix for Random Forest:\n", conf_matrix_rf)

# Perform predictions on the test set
y_pred_rf = RF.predict(X_test_classification)
y_prob_rf = RF.predict_proba(X_test_classification)[:, 1]

# Confusion Matrix
conf_matrix_rf = confusion_matrix(y_true=y_test_classification, y_pred=y_pred_rf)
print("Confusion Matrix for Random Forest:\n", conf_matrix_rf)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Random Forest)')
plt.show()

# ROC Curve
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test_classification, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'AUC = {roc_auc_rf:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Random Forest)')
plt.legend()
plt.show()

# AUC Score
print("AUC Score for Random Forest:", roc_auc_rf)


## Comparing 4 models (except SVM) ROC and AUC

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

models = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"]
y_probs = {
    "Logistic Regression": final_model.predict_proba(X_test_scaled_final)[:, 1],
    "KNN": knn_classification_final.predict_proba(X_test_classification_tran)[:, 1],
    "Decision Tree": clf_grid_search.predict_proba(X_test_classification)[:, 1],
    "Random Forest": RF.predict_proba(X_test_classification)[:, 1],
}

# Plot ROC curves
plt.figure(figsize=(8, 8))

for model in models:
    fpr, tpr, thresholds = roc_curve(y_test_classification, y_probs[model], pos_label=1)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model} (AUC = {auc_score:.2f}')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend()
plt.show()