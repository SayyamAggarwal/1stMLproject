import os 
import numpy as np
import pandas as pd
import sys
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV


def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train,Y_train,X_test,Y_test,models,param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = list(param.values())[i] 
            gs = GridSearchCV(model,para,cv = 5,n_jobs=-1)
            # model.fit(X_train,Y_train)
            gs.fit(X_train,Y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,Y_train)

            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)



            train_score = r2_score(Y_train,Y_train_pred)
            test_score = r2_score(Y_test,Y_test_pred)
            report[list(models.keys())[i]] = test_score
             
        return report
    except Exception as e:
        raise CustomException(e,sys)
    

def load_obj(file_name):
    try:
        with open(file_name,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    