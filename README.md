# MLOps Experiment Tracking with MLflow & DagsHub

This project demonstrates a complete MLOps workflow for tracking machine learning experiments, hyperparameter tuning, and model versioning using **MLflow** and **DagsHub**.

## 🚀 Overview
The repository contains scripts to train a Random Forest model, perform Grid Search for hyperparameter optimization, and log all metrics, parameters, and models to a remote tracking server.

## 🛠️ Tech Stack
- **Language:** Python 3.12+
- **ML Framework:** Scikit-Learn
- **Experiment Tracking:** MLflow
- **Remote Storage:** DagsHub (Hosted MLflow)

## 📁 Project Structure
```text
MLOps-EXPERIMENTS-TRACKING-WITH-MLFLOW/
├── src/
│   ├── file1.py           # Basic MLflow logging (Local)
│   ├── file2.py           # DagsHub integration setup
│   └── hypertune1.py      # Grid Search & Autologging implementation
├── requirements.txt       # Project dependencies
└── README.md              # Documentation
```

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd MLOps-EXPERIMENTS-TRACKING-WITH-MLFLOW
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **DagsHub Authentication:**
   Run the following command to connect your local environment to DagsHub:
   ```bash
   dagshub login
   ```

## 📊 Running Experiments

### Hyperparameter Tuning
To run the Random Forest hyperparameter tuning script (Grid Search):
```bash
python src/hypertune1.py
```
This script will:
- Test 9 combinations of `n_estimators` and `max_depth`.
- Log the **Best Parameters**: `{'max_depth': None, 'n_estimators': 200}`.
- Log the **Best Accuracy**: `~0.9626`.
- Automatically log the model schema and artifacts to DagsHub.

## 📈 Results
Once the script finishes, you can view the experiment results, compare runs, and visualize the parameter-metric relationships on your [DagsHub Experiment Dashboard](https://dagshub.com).

---
*Developed as part of the MLOps Learning Path.*
