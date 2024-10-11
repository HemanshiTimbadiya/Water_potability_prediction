# Water_potability_prediction

This repository contains the code and resources for predicting water potability based on various chemical and physical water quality parameters. The goal of the project is to determine whether water is safe for human consumption using machine learning models trained on a water quality dataset.

 # Project Overview
Access to clean and safe drinking water is crucial for human health. This project focuses on predicting whether water is Potable (safe to drink) or Not Potable based on water quality parameters such as pH, Hardness, Solids, Chloramines, Sulfates, and more. The machine learning models developed in this project aim to automate the process of water quality assessment.


## Dataset
The dataset contains the following features (input parameters):

pH: Measure of the acidity or basicity of the water (WHO standard: 6.5–8.5)

Hardness: Measure of calcium and magnesium ions in water (WHO standard: 200 mg/L)

Solids (TDS): Total dissolved solids, indicates mineral content (WHO standard: 1000 ppm)

Chloramines: Used for water disinfection (WHO standard: up to 4 ppm)

Sulfate: Naturally occurring in water, affects taste (WHO standard: 1000 mg/L)

Conductivity: Measure of water’s ability to conduct electricity (WHO standard: 400 μS/cm)

Organic Carbon: Indicator of organic pollutants (WHO standard: 10 ppm)

Trihalomethanes: By-product of water chlorination (WHO standard: 80 ppm)

Turbidity: Measure of water clarity (WHO standard: 5 NTU)

The target variable is:

Potability: Whether the water is safe to drink or not (1 for Potable, 0 for Not Potable)
Machine Learning Models
Several machine learning algorithms are used to classify water as Potable or Not Potable, including:

# Machine Learning Models
Several machine learning algorithms are used to classify water as Potable or Not Potable, including:

K-Nearest Neighbors (KNN)
Support Vector Classifier (SVC)
Decision Tree
Random Forest
XGBoost
The dataset was preprocessed by handling missing values, balancing the class distribution using SMOTE (Synthetic Minority Over-sampling Technique), and performing feature scaling where necessary.


 # Model Evaluation
Each model is evaluated based on its accuracy, precision, recall, and F1-score using cross-validation. Hyperparameter tuning was performed to optimize model performance.




