import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__ (self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_transformer_obj(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            num_features = ['reading score', 'writing score']
            cat_features = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
                ]
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ("Onehotencoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )


            logging.info(f"num_features :{num_features}")
            logging.info(f"cat_features :{cat_features}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipe",num_pipeline,num_features),
                    ("cat_pipe",cat_pipeline,cat_features)
                ]
            )

            return preprocessor
        

        except Exception as e:
            raise CustomException(e,sys)
        


    def init_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data")

            logging.info("obtaining preprocessor obj")
            preprocessor_obj = self.get_transformer_obj()
            target_column_name = 'math score'
            num_features = ['reading score', 'writing score']
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis = 1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessor on training and testing dataset ")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessor obj")

            save_obj(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
                
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



        except Exception as e:
            raise CustomException(e,sys)




















