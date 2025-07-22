## We need to import each and every algorithm
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj,evaluate_models

@dataclass 
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def init_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting TrainTest Input data")
            X_train,X_test,Y_train,Y_test = (
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'SVR': SVR(),
                'XGBRegressor': XGBRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=0)
            }

            model_report:dict = evaluate_models(X_train=X_train,
                                               Y_train = Y_train,
                                               X_test=X_test,
                                               Y_test=Y_test,
                                               models = models) 


            best_model_score = max(sorted(model_report.values())) 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]    

            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best Model found",sys)      
            logging.info(f"Best Model Found: {best_model_name} with score: {best_model_score}")
            save_obj(file_path=self.model_trainer_config.trained_model_path,
                     obj=best_model)
            
            logging.info(f"Model saved at {self.model_trainer_config.trained_model_path}")
            return best_model_score
        except Exception as e:
            raise CustomException(e,sys)












