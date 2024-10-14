# Water Potability Prediction

## Project Overview
Access to clean and safe drinking water is crucial for human health. This project focuses on predicting whether water is Potable (safe to drink) or Not Potable based on water quality parameters such as pH, Hardness, Solids, Chloramines, Sulfates, and more. The machine learning models developed in this project aim to automate the process of water quality assessment.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Machine Learning Models](#machine-learning-models)
- [Model Evaluation](#model-evaluation)
- [Installation & Usage](#installation--usage)

## Technologies Used
| Technology         | Description                               |
|--------------------|-------------------------------------------|
| Python             | Core programming language.                |
| Pandas             | Data manipulation and analysis.           |
| Matplotlib & Seaborn | Data visualization.                     |
| Scikit-learn       | Clustering algorithms.                     |
| Streamlit          | Web app framework to deploy the project. |

## Project Structure
The project files and their purposes are as follows:

| File/Directory                     | Purpose                                                         |
|------------------------------------|-----------------------------------------------------------------|
| `water_potability`                 | Contains the analysis and model training dataset.              |
| `Water potability Prediction`       | Jupyter notebooks with Exploratory Data Analysis (EDA), feature engineering, and model implementation. |
| `EDA plots`                        | Visualizations and graphs.                                      |
| `README.md`                        | Project documentation and instructions.                        |
| `requirements.txt`                 | List of required libraries and dependencies to run the project.|
| `water_main.py`                   | Contains the saved SVC model for deployment.                  |
| `water_app.py`                    | Streamlit application for Water potability.                   |

## Dataset
The dataset contains the following features (input parameters):

| Feature              | Description                                                | WHO Standard                 |
|----------------------|------------------------------------------------------------|-------------------------------|
| **pH**               | Measure of the acidity or basicity of the water           | 6.5–8.5                      |
| **Hardness**         | Measure of calcium and magnesium ions in water            | 200 mg/L                     |
| **Solids (TDS)**     | Total dissolved solids, indicates mineral content         | 1000 ppm                     |
| **Chloramines**      | Used for water disinfection                               | Up to 4 ppm                  |
| **Sulfate**          | Naturally occurring in water, affects taste              | 1000 mg/L                    |
| **Conductivity**     | Measure of water’s ability to conduct electricity         | 400 μS/cm                    |
| **Organic Carbon**   | Indicator of organic pollutants                            | 10 ppm                       |
| **Trihalomethanes**  | By-product of water chlorination                          | 80 ppm                       |
| **Turbidity**        | Measure of water clarity                                   | 5 NTU                        |

The target variable is:
- **Potability**: Whether the water is safe to drink or not (1 for Potable, 0 for Not Potable).

## Machine Learning Models
Several machine learning algorithms are used to classify water as Potable or Not Potable, including:
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Decision Tree
- Random Forest
- XGBoost

The dataset was preprocessed by handling missing values, balancing the class distribution using SMOTE (Synthetic Minority Over-sampling Technique), and performing feature scaling where necessary.

## Model Evaluation
Each model is evaluated based on its accuracy, precision, recall, and F1-score using cross-validation. Hyperparameter tuning was performed to optimize model performance.

## Installation & Usage

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/HemanshiTimbadiya/Water_potability_prediction.git
2. Navigate to the project directory:
   ```bash
   cd Water potability prediction
   
4. Install the required packages:
   ```bash
   pip install -r requirements.txt

### Usage
- Run the analysis in a Jupyter Notebook.
- To deploy the project using Streamlit, run:
  ```bash
  streamlit run water_app.py 
