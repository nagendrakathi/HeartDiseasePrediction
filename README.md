# Heart Disease Prediction using XGBoost

This project implements a machine learning model to predict heart disease using the XGBoost classifier. The dataset used contains health-related parameters to classify the presence or absence of heart disease.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Code Explanation](#code-explanation)
- [Results](#results)
- [Streamlit Application](#streamlit-application)
---

## Project Overview
The objective of this project is to:
1. Load and preprocess the heart disease dataset.
2. Train a machine learning model using XGBoost.
3. Evaluate the model using performance metrics like confusion matrix, classification report, and accuracy score.
4. Save the trained model for deployment or future use.

---

## Dataset
- The dataset should be in CSV format and named `Heart_Disease_Prediction.csv`.
- Ensure the dataset contains a target column named `Heart Disease` along with relevant features.
- Missing values in the dataset are handled using forward fill (`ffill`).

---

## Requirements
The following Python libraries are required to run this project:

- pandas
- numpy
- scikit-learn
- xgboost
- joblib

You can install them using the following command:
```bash
pip install pandas numpy scikit-learn xgboost joblib
```

---

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/HeartDiseasePrediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd HeartDiseasePrediction
   ```

3. Place the dataset file `Heart_Disease_Prediction.csv` in the project directory.

4. Run the Python script:
   ```bash
   python heart_disease_prediction.py
   ```

5. The trained model will be saved in the directory `Model` as `xgboost_model.pkl`.

---

## Code Explanation

### 1. Load and Inspect Data
- Load the dataset using pandas.
- Check for missing values and handle them using forward fill (`ffill`).
- Encode the target variable (`Heart Disease`) using `LabelEncoder`.

### 2. Split Data
- Split the dataset into features (`X`) and target (`y`).
- Further split the data into training and testing sets using an 80-20 ratio.

### 3. Train the Model
- Train an XGBoost classifier using the training data.
- Use the `eval_metric` parameter set to `logloss` for proper evaluation.

### 4. Evaluate the Model
- Use the test set to predict outcomes.
- Evaluate performance using confusion matrix, classification report, and accuracy score.

### 5. Save the Model
- Save the trained model as a `.pkl` file using `joblib` for future use.

---

## Results
- The trained model is evaluated on various metrics, including:
  - Confusion Matrix
  - Classification Report
  - Accuracy Score

Example output:
```
Confusion Matrix:
[[50  5]
 [ 8 37]]

Classification Report:
              precision    recall  f1-score   support

           0       0.86      0.91      0.88        55
           1       0.88      0.82      0.85        45

    accuracy                           0.87       100
   macro avg       0.87      0.86      0.87       100
weighted avg       0.87      0.87      0.87       100

Accuracy Score: 0.87
```

## Streamlit Application
The Streamlit app provides an intuitive interface for users to input their health parameters and get predictions.

### Features:
1. **Input Fields**: Enter details such as age, sex, chest pain type, blood pressure, cholesterol, and more.
2. **Predict Button**: Generate predictions for heart disease based on the input.
3. **Reset Button**: Clear all input fields for new entries.
4. **Prediction Display**: Results are shown in green if no heart disease is detected or red if heart disease is detected.

### Inputs Required:
- Age
- Sex (Male/Female)
- Chest pain type
- Blood Pressure
- Cholesterol
- Fasting Blood Sugar (FBS over 120)
- EKG Results
- Max Heart Rate
- Exercise Angina
- ST Depression
- Slope of ST
- Number of Vessels Fluro
- Thallium

---

## Results
- The trained model is integrated into the Streamlit app, providing real-time predictions.
- Example outputs:
  - **No Heart Disease Detected**: 
    ![Green Output Example](example_no_disease.png)
  - **Heart Disease Detected**: 
    ![Red Output Example](example_disease_detected.png)
---

## Contributing
Feel free to fork this repository and contribute by submitting pull requests. Suggestions and improvements are welcome!

---

## Acknowledgments
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Dataset](https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction)