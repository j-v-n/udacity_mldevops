CATEGORY_LIST = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

KEEP_COLS = ["Customer_Age",
             "Dependent_count",
             "Months_on_book",
             "Total_Relationship_Count",
             "Months_Inactive_12_mon",
             "Contacts_Count_12_mon",
             "Credit_Limit",
             "Total_Revolving_Bal",
             "Avg_Open_To_Buy",
             "Total_Amt_Chng_Q4_Q1",
             "Total_Trans_Amt",
             "Total_Trans_Ct",
             "Total_Ct_Chng_Q4_Q1",
             "Avg_Utilization_Ratio"]

TEST_SIZE = 0.3
RANDOM_STATE = 42
LR_SOLVER = 'lbfgs'
LR_MAX_ITER = 3000

PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}
