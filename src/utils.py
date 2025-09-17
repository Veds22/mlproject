import sys
import os

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging 

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        
        for model_technique, model in models.items():
            model_obj = model
            param = params[model_technique]
            
            gs = GridSearchCV(model, param, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)
             
            model_obj.set_params(**gs.best_params_)
            model_obj.fit(X_train, y_train)
            
            y_train_pred = model_obj.predict(X_train)
            y_test_pred = model_obj.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[model_technique] = {
                'train model score': train_model_score,
                'test model score': test_model_score,
                'best params': gs.best_params_
            }
            
        return report
            
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)    