# library doc string
"""
Library of functions to predict customers who are likely to churn

Author - Jayanth Nair (Nov 13, 2022)
"""

# import basic libraries
import os
import joblib
# import ds specific libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
# import sklearn models
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# import constants
import constants

sns.set()

os.environ["QT_QPA_PLATFORM"] = "offscreen"


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """

    df = pd.read_csv(pth)
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    return df


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    # create churn plot
    plt.figure(figsize=(20, 10))
    df["Churn"].hist()
    plt.savefig("./images/eda/churn.png")
    # create customer age plot
    plt.figure(figsize=(20, 10))
    df["Customer_Age"].hist()
    plt.savefig("./images/eda/age.png")
    # create marital status plot
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.savefig("./images/eda/marital_status.png")
    # create trans ct plot
    plt.figure(figsize=(20, 10))
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    plt.savefig("./images/eda/total_trans_ct.png")
    # create heat map
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig("./images/eda/heatmap.png")


def encoder_helper(df, category_lst, response="Churn"):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be 
            used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    for category in category_lst:
        category_groups = df.groupby(category).mean()[response]
        blank_lst = [category_groups.loc[val] for val in df[category]]
        df[category + "_" + response] = blank_lst
    return df


def perform_feature_engineering(df, response="Churn"):
    """
    input:
              df: pandas dataframe
              response: string of response name 
              [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    keep_cols = constants.KEEP_COLS.copy()
    keep_cols.extend(["Gender_" + response,
                      "Education_Level_" + response,
                      "Marital_Status_" + response,
                      "Income_Category_" + response,
                      "Card_Category_" + response,
                      ])

    X = pd.DataFrame()
    y = df[response]
    X[keep_cols] = df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=constants.TEST_SIZE, random_state=constants.RANDOM_STATE)
    return X_train, X_test, y_train, y_test


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    plt.style.use("classic")
    _, axs = plt.subplots(4, figsize=(10, 15))

    for count, ax in enumerate(axs):
        if count == 0:
            truth, preds = y_test, y_test_preds_rf
            plot_title = "Random Forest Test Results"
        elif count == 1:
            truth, preds = y_train, y_train_preds_rf
            plot_title = "Random Forest Train Results"
        elif count == 2:
            truth, preds = y_test, y_test_preds_lr
            plot_title = "Logistic Regression Test Results"
        elif count == 3:
            truth, preds = y_train, y_train_preds_lr
            plot_title = "Logistic Regression Train Results"

        # save classification report of the test results of random forest
        # classifier
        ax.text(
            0.1,
            0.5,
            str(classification_report(truth, preds)),
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        # txt.set_clip_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        ax.set_title(plot_title, loc="left")
    # plt.tight_layout()
    plt.savefig("./images/results/classification_reports.png")
    # plt.show()


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    plt.figure(figsize=(20, 5))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_pth + "/shap_file.png")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    names = [X_data.columns[i] for i in indices]
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    #   Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(output_pth + "/feature_importances.png")


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    rfc = RandomForestClassifier(random_state=constants.RANDOM_STATE)
    lrc = LogisticRegression(solver=constants.LR_SOLVER,
                             max_iter=constants.LR_MAX_ITER)

    param_grid = constants.PARAM_GRID

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
    feature_importance_plot(cv_rfc.best_estimator_,
                            X_test, "./images/results/")

    # save models
    joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
    joblib.dump(lrc, "./models/logistic_model.pkl")


if __name__ == "__main__":
    data_frame = import_data("./data/bank_data.csv")

    category_list = constants.CATEGORY_LIST
    perform_eda(data_frame)
    data_frame = encoder_helper(data_frame, category_list, response="Churn")
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        data_frame, "Churn")
    # train models, save figures and models
    train_models(X_train=X_train, X_test=X_test,
                 y_train=y_train, y_test=y_test)
