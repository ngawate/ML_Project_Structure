import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import ColumnTransformer

page_name = 'Model_Trainer.py'

@dataclass
class Model_Trainer_Config:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class Model_Trainer:
    def __init__(self):
        self.model_trainer_config = Model_Trainer_Config()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info(f'Train Test Split -> {page_name}')
            X_train, y_train, X_test, y_test = (train_arr[:,:-1], train_arr[:,-1], test_arr[:,:-1], test_arr[:,-1])

            models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "AdaBoost": AdaBoostRegressor(),
            "SVR": SVR(),
            "XGBoost": XGBRegressor(),
            "KNeighbors": KNeighborsRegressor()
            }

            model_report: dict=evaluate_model(X_train=X_train,y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            best_model_score = max(list(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            logging.info(f'Got the best model --> {best_model_name} --> {page_name}')

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj = best_model_name)

            predicted = best_model_name.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return best_model_name, r2_square

        except Exception as e:
            logging.error(f'Error in Model Training --> {page_name}')
            raise CustomException(e, sys)