import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report , roc_curve, roc_auc_score

data =pd.read_csv("C:/Users/HP\Downloads/water_potability.csv")
data.head()

data.dtypes

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Handle missing values using median
data.fillna(data.median(), inplace=True)

# Feature engineering
#data['pH_Hardness'] = data['ph'] * data['Hardness']
data.head()

# Separate features (X) and target (y)
X = data.drop('Potability', axis=1)
y = data['Potability']

# Initialize StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance data using SMOTE.

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

#  Print the class distribution after SMOTE
print("Class distribution after SMOTE:")
print(y_resampled.value_counts())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled,y_resampled, test_size=0.2, random_state=42)

svc = SVC(random_state=42)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of SVC Base model:", accuracy)


# Define a more extensive parameter grid for SVC
param_grid_svc = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],  # Multiple kernel options
    'gamma': ['scale', 'auto', 0.1, 1, 10],  # Various gamma options
}

# Initialize the SVC classifier
svc = SVC(random_state=42)

# Initialize the Random Search with cross-validation for SVC
random_search_svc = RandomizedSearchCV(estimator=svc,
                                       param_distributions=param_grid_svc,
                                       n_iter=20,  # Increased iterations for better coverage
                                       cv=5,  # Slightly increased folds for better estimation
                                       scoring='accuracy',
                                       random_state=42,
                                       n_jobs=-1)

# Fit the model to find the best hyperparameters
random_search_svc.fit(X_train, y_train)

# Get the best hyperparameters
best_params_svc = random_search_svc.best_params_
# print("Best Hyperparameters for SVC:", best_params_svc)

# Create and train the best SVC model
best_svc = SVC(**best_params_svc, random_state=42)
best_svc.fit(X_train, y_train)

# Evaluate the best model on the test set
y_pred_best_svc = best_svc.predict(X_test)
accuracy_best_svc = accuracy_score(y_test, y_pred_best_svc)
print("Accuracy of SVC Model After Hyperparameter Tuning ):", accuracy_best_svc)

cm = confusion_matrix(y_test, y_pred_best_svc)
print("Confusion Matrix (Best SVC Model):\n", cm)


# Classification Report
cr = classification_report(y_test, y_pred_best_svc)
print("Classification Report:\n", cr)

# Save the best SVC model and the scaler
joblib.dump(best_svc, 'best_SVC_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")





