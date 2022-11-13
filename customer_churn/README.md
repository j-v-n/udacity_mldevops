# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project predicts customer churn based on provided bank data

## Files and data description
```
📦customer_churn
 ┣ 📂data (contains the data we train the model on)
 ┃ ┗ 📜bank_data.csv
 ┣ 📂images
 ┃ ┣ 📂eda (contains the results of the eda)
 ┃ ┃ ┣ 📜age.png
 ┃ ┃ ┣ 📜churn.png
 ┃ ┃ ┣ 📜heatmap.png
 ┃ ┃ ┣ 📜marital_status.png
 ┃ ┃ ┗ 📜total_trans_ct.png
 ┃ ┗ 📂results (contains the results from modeling)
 ┃ ┃ ┣ 📜classification_reports.png
 ┃ ┃ ┣ 📜feature_importances.png
 ┃ ┃ ┗ 📜shap_file.png
 ┣ 📂logs (contains the logs from running the tests)
 ┃ ┗ 📜churn_library.log
 ┣ 📂models (contains the saved model files)
 ┃ ┣ 📜logistic_model.pkl
 ┃ ┗ 📜rfc_model.pkl
 ┣ 📜Guide.ipynb
 ┣ 📜README.md
 ┣ 📜churn_library.py (contains the functions to do EDA and model training)
 ┣ 📜churn_notebook.ipynb (preliminary notebook which has been refactored into churn_library.py)
 ┣ 📜churn_script_logging_and_tests.py (contains tests done on churn_library.py)
 ┣ 📜constants.py (contains constants used in EDA and model training)
 ┣ 📜requirements_py3.6.txt
 ┗ 📜requirements_py3.8.txt
```
## Running Files
To do EDA, train model and save model files and results

```
python churn_library.py
```

To run the tests

```
python churn_script_logging_and_tests.py
```

