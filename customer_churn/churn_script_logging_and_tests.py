"""
Script to run tests on churn_library.py

Author - Jayanth Nair (Nov 13, 2022)
"""

import os
import logging
import churn_library as cls
import constants

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("SUCCESS: Testing import_data")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function - checks for if expected images are saved
    '''
    # directories for images and data
    images_pth = "./images/eda"
    data_pth = "./data/bank_data.csv"
    fnames = ["age.png", "churn.png", "heatmap.png",
              "marital_status.png", "total_trans_ct.png"]
    # load data
    df = cls.import_data(data_pth)
    # do eda
    cls.perform_eda(df)
    # check if expected image files are saved
    saved_images = os.listdir(images_pth)
    try:
        for fname in fnames:
            assert fname in saved_images
            logging.info(f'filename {str(fname)} exists')
            assert os.path.getsize(os.path.join(images_pth, fname)) > 0
            logging.info(f'filename {str(fname)} is not a blank image')
    except AssertionError as err:
        logging.error(
            'perform_eda not working as expected. a file has not been saved')
        raise err

    logging.info('SUCCESS: perform_eda working as expected')


def test_encoder_helper():
    '''
    test encoder helper - checks if required columns exist
    '''
    # data path
    data_pth = "./data/bank_data.csv"
    # load data
    df = cls.import_data(data_pth)
    # do encoding
    category_list = constants.CATEGORY_LIST
    df = cls.encoder_helper(df, category_list, response="Churn")

    try:
        for cat in category_list:
            assert str(f"{cat}_Churn") in df.columns
            logging.info(f'column {str(cat)}_Churn exists')
    except AssertionError as err:
        logging.error(
            'encoder_helper function does not create all required columns')
        raise err

    logging.info('SUCCESS: encoder_helper working as expected')


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    # data path
    data_pth = "./data/bank_data.csv"
    # load data
    df = cls.import_data(data_pth)
    # do encoding
    category_list = constants.CATEGORY_LIST
    df = cls.encoder_helper(df, category_list, response="Churn")
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        df, "Churn")
    try:
        assert (X_train.shape[0] > 0) & (X_train.shape[1] > 0)
        logging.info('X_train is verified to not be empty')
        assert (X_test.shape[0] > 0) & (X_test.shape[1] > 0)
        logging.info('X_test is verified to not be empty')
        assert y_train.shape[0] == X_train.shape[0]
        logging.info('y_train is verified to have same # of rows as X_train')
        assert y_test.shape[0] == X_test.shape[0]
        logging.info('y_test is verified to have same # of rows as X_test')
    except AssertionError as err:
        logging.error(
            'The resultant dataframes are not of the expected shape and size')
        raise err

    logging.info("SUCCESS: perform_feature_engineering is working as expected")


def test_train_models():
    '''
    test train_models
    '''
    # models path
    models_path = "./models/"
    # images path
    images_path = "./images/results/"
    # data path
    data_pth = "./data/bank_data.csv"
    # load data
    df = cls.import_data(data_pth)
    # do encoding
    category_list = constants.CATEGORY_LIST
    df = cls.encoder_helper(df, category_list, response="Churn")
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        df, "Churn")
    cls.train_models(X_train, X_test, y_train, y_test)
    saved_model_files = os.listdir(models_path)

    model_files = ["logistic_model.pkl", "rfc_model.pkl"]
    # check if model files are being saved
    try:
        for model in model_files:
            assert model in saved_model_files
            logging.info(f"{model} has been verified to be saved")
    except AssertionError as err:
        logging.error("Required model files not saved")
        raise err
    logging.info("SUCCESS: Expected model files are being saved")

    # check if image files are being saved
    saved_image_files = os.listdir(images_path)
    image_files = ["feature_importances.png", "shap_file.png"]
    try:
        for image in image_files:
            assert image in saved_image_files
            logging.info(f"{image} has been verified to be saved")
    except AssertionError as err:
        logging.error("Required results files not saved")
        raise err
    logging.info("SUCCESS: Expected images files are being saved")
    logging.info(
        "SUCCESS: Models are being trained, weights and results have been saved ")


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
