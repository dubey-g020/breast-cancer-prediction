# Breast Cancer Prediction Using Machine Learning

## Overview

This project is a Machine Learning based web application that predicts whether a breast tumor is **Benign** or **Malignant** using medical diagnostic features.

The application is built using:

* Python
* Flask
* Scikit-learn
* Bootstrap
* Logistic Regression

The trained ML model is integrated with a Flask web application to provide real-time predictions through a user-friendly interface.

---

# Features

* Breast cancer prediction using Machine Learning
* Logistic Regression classification model
* Data preprocessing and feature scaling
* Flask web application
* Bootstrap responsive frontend
* Real-time prediction system
* Scalable deployment-ready architecture

---

# Technologies Used

| Technology   | Purpose              |
| ------------ | -------------------- |
| Python       | Programming Language |
| Flask        | Backend Framework    |
| Scikit-learn | Machine Learning     |
| Pandas       | Data Handling        |
| NumPy        | Numerical Operations |
| Bootstrap    | Frontend Styling     |
| Pickle       | Model Serialization  |

---

# Machine Learning Workflow

1. Dataset Loading
2. Data Preprocessing
3. Train-Test Split
4. Feature Scaling using StandardScaler
5. Model Training using Logistic Regression
6. Model Evaluation
7. Model Saving using Pickle
8. Flask Integration
9. Web Deployment

---

# Dataset

Dataset used:

* Breast Cancer Wisconsin Dataset

Source:
[https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

---

# Project Structure

```bash
breast_cancer_app/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-prediction.git
```

## Open Project Folder

```bash
cd breast-cancer-prediction
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Application

```bash
python app.py
```

Open browser:

```bash
http://127.0.0.1:5000
```

---

# Model Training

The model was trained using:

```python
LogisticRegression(max_iter=1000)
```

Feature scaling was performed using:

```python
StandardScaler()
```

---

# Prediction Classes

| Value | Meaning          |
| ----- | ---------------- |
| 0     | Malignant Cancer |
| 1     | Benign Cancer    |

---

# Screenshots

Add screenshots of:

* Home Page
* Prediction Result
* Input Form

---

# Future Enhancements

* Prediction confid
