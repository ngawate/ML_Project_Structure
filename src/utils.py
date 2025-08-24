import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

page_name = 'utils'

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train ,y_train, X_test, y_test, models):
    try:
        report = {}

        for name, model in models.items():

            model.fit(X_train, y_train)
        
            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            report[model] = test_model_score

        return report
    
    except Exception as e:
        logging.error(f'{e} -- >{page_name}')
        raise CustomException(e, sys)
    