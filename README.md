# Breast Cancer Detection using SVM

A machine learning project that classifies breast tumors as **Malignant** or **Benign** using a Support Vector Machine (SVM) model trained on the Wisconsin Breast Cancer dataset.

## 📌 Overview

This project uses a Support Vector Machine classifier to predict whether a breast mass is cancerous based on features computed from a digitized image of a fine needle aspirate (FNA) of the breast mass. The entire pipeline — data loading, preprocessing, training, and evaluation — is implemented in a single Jupyter Notebook.

## 📂 Project Structure

```
breast-cancer-detection/
│
├── breast_cancer_detection.ipynb   # Main notebook (data prep, training, evaluation)
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

## 📊 Dataset

- **Source:** Wisconsin Breast Cancer Diagnostic Dataset (available via `sklearn.datasets.load_breast_cancer()` or the [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic))
- **Samples:** 569
- **Classes:** Malignant (0), Benign (1)
- **Features:** 30 numeric features computed from digitized FNA images

## 🔑 Feature (Update) Keys

These are the dataset's feature keys/columns — use these exact names if updating, filtering, or engineering features in the notebook:

**Mean features:**
`mean radius`, `mean texture`, `mean perimeter`, `mean area`, `mean smoothness`, `mean compactness`, `mean concavity`, `mean concave points`, `mean symmetry`, `mean fractal dimension`

**Standard error (SE) features:**
`radius error`, `texture error`, `perimeter error`, `area error`, `smoothness error`, `compactness error`, `concavity error`, `concave points error`, `symmetry error`, `fractal dimension error`

**"Worst" (largest mean) features:**
`worst radius`, `worst texture`, `worst perimeter`, `worst area`, `worst smoothness`, `worst compactness`, `worst concavity`, `worst concave points`, `worst symmetry`, `worst fractal dimension`

**Target key:**
`target` → 0 = malignant, 1 = benign

> When updating the model with new data, ensure your input matches these 30 feature keys exactly, in the same order, and is scaled using the same scaler fitted on the training data.

## ⚙️ Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
```

Install with:
```bash
pip install -r requirements.txt
```

## 🚀 How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/uday-tc/breast-cancer-detection.git
   cd breast-cancer-detection
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the notebook
   ```bash
   jupyter notebook breast_cancer_detection.ipynb
   ```
4. Run all cells to reproduce data preprocessing, training, and evaluation.

## 🧠 Model

- **Algorithm:** Support Vector Machine (SVM)
- **Preprocessing:** Feature scaling (StandardScaler) applied to all 30 features before training
- **Train/Test Split:** 80/20 (adjust as per your notebook's actual split)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

## 📈 Results

| Metric      | Score |
|-------------|-------|
| Accuracy    |  |
| Precision   |  |
| Recall      |  |
| F1-score    |  |

*(Fill in with the actual metrics from your notebook's output.)*

## 🔮 Future Improvements

- Compare SVM against other models (Logistic Regression, Random Forest) for benchmarking
- Add hyperparameter tuning (GridSearchCV) for the SVM kernel and C/gamma values
- Build a simple Streamlit/Flask UI for live predictions
- Add cross-validation for more robust performance estimates

## 👤 Author

**Udaya Shankar T C**
- GitHub: [uday-tc](https://github.com/uday-tc)
- LinkedIn: [uday03](https://linkedin.com/in/uday03)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
