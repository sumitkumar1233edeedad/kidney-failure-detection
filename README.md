# 🫘 Kidney Failure Detection — ML + Streamlit App

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

> A machine learning project to detect **Chronic Kidney Disease (CKD)**
> using patient clinical data — with an interactive Streamlit web app
> for prediction and data exploration.

---

## 🖥️ Live Demo
👉 [Open Streamlit App](https://your-app-link.streamlit.app)

---

## 📌 About the Project

Chronic Kidney Disease (CKD) is a silent disease that often goes
undetected until it reaches an advanced stage. This project aims to:

- Analyze clinical features related to kidney failure
- Build a machine learning model to predict CKD
- Deploy an interactive web app for real-time prediction

---

## 📊 Dataset

| Detail | Info |
|---|---|
| Source | [Kaggle](https://www.kaggle.com/datasets/mansoordaku/ckdisease) |
| Rows | 400 patients |
| Columns | 25 features |
| Target | CKD / Not CKD |

**Key Features Used:**
- Blood Pressure, Specific Gravity, Albumin
- Sugar, Red Blood Cells, Serum Creatinine
- Hemoglobin, Packed Cell Volume, GFR
- Hypertension, Diabetes Mellitus, Appetite

---

## 🏗️ Project Structure
```
kidney-failure-detection/
│
├── data/
│                   # Cleaned dataset
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_model_building.ipynb
├── app/
│   ├── app.py                      # Streamlit main app
│   
│
├── outputs/
│   ├── figures/
│   └── models/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data processing |
| Matplotlib & Seaborn | Visualization |
| Scikit-learn | ML model building |
| Plotly | Interactive charts |
| Streamlit | Web app deployment |
| Joblib | Model saving/loading |
| GitHub | Version control |

---

## 🚀 Run Locally
```bash
# 1. Clone the repo
git clone https://github.com/your-username/kidney-failure-detection.git

# 2. Navigate into the folder
cd kidney-failure-detection

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app/app.py
```

---

## 📈 ML Models Used

- [ ] naive_bayes(GaussianNB)
- [ ] Decision Tree

> Best model selected based on Accuracy, F1-Score & ROC-AUC

---

## 🖼️ App Pages

| Page | Description |
|---|---|
| 🏠 Home | Project intro and summary |
| 📊 Data Overview | Dataset preview and statistics |
| 📉 EDA | Charts, heatmaps, distributions |
| 🔍 Prediction | Enter patient data → get result |
| 📋 Model Performance | Accuracy, confusion matrix, ROC |

---

## 👤 Author

| Detail | Info |
|---|---|
| **Name** | Sumit Kumar |
| **Course** | BCA |
| **Institution** | SVSD BHATOLI |
| **GitHub** | [@vanshu](https://github.com/sumitkumar1233edeedad) |

---

## 📖 References

- UCI ML Repository — CKD Dataset
- WHO — Chronic Kidney Disease Guidelines
- Scikit-learn Documentation
- Streamlit Documentation

---

## 📜 License

This project is licensed under the
[MIT License](./LICENSE) — free to use for academic purposes.

---

⭐ If you found this project helpful, please give it a star!