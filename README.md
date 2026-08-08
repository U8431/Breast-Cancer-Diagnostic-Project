# Breast Cancer Detection

A machine learning pipeline that classifies breast tumor diagnostic data as **malignant** or **benign**, built during a Data Science internship at Take It Smart (OPC) Pvt Ltd (Sep 2025 – Dec 2025).

## Overview

This project applies the full data science workflow — cleaning, exploration, dimensionality reduction, and model training/comparison — to a breast cancer diagnostic dataset, with the goal of predicting tumor malignancy from measurements derived from digitized cell nuclei images.

## Tech Stack

- **Python**
- **Pandas / NumPy** — data handling and preprocessing
- **Matplotlib / Seaborn** — exploratory data analysis and visualization
- **Scikit-learn** — model training and evaluation
- **PCA** — dimensionality reduction

## Pipeline

1. **Data Cleaning**
   Checked for missing values and irrelevant columns, and converted the diagnosis label into numeric form for model training.

2. **Exploratory Data Analysis (EDA)**
   Visualized class distribution and feature relationships using Seaborn/Matplotlib, including correlation heatmaps and distribution comparisons between malignant and benign cases.

3. **Dimensionality Reduction (PCA)**
   Applied Principal Component Analysis to reduce the feature space, addressing high correlation between related measurements (e.g. radius, perimeter, area) and improving model efficiency.

4. **Model Training**
   Trained and compared two classifiers:
   - **Random Forest** — an ensemble of decision trees, robust to non-linear relationships
   - **Support Vector Machine (SVM)** — finds the optimal separating hyperplane between classes

5. **Evaluation**
   Compared models using accuracy, precision/recall, F1-score, and confusion matrix, with particular attention to recall on the malignant class, since missing a malignant case (false negative) carries a much higher cost than a false positive in a medical context.

## Results

Both models achieved strong classification performance, with SVM and Random Forest each showing high accuracy in distinguishing malignant from benign cases.

## Key Learnings

- Handling correlated features in medical diagnostic data
- Applying PCA for dimensionality reduction without losing predictive signal
- Choosing evaluation metrics appropriate to the domain — prioritizing recall over raw accuracy in a healthcare context
- Comparing ensemble methods (Random Forest) against margin-based classifiers (SVM)

## Author

**Udaya Shankar T C**
[LinkedIn](https://www.linkedin.com/in/uday03/) · [GitHub](https://github.com/uday-tc)
