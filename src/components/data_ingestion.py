import os
import pandas as pd
import sys 
from src.logger import logging
import src.exception as CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","data.csv")



# Start


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def init_data_ingestion(self):
        logging.info("Entered data ingestion method or component")
        try:
            df = pd.read_csv("notebook\\StudentsPerformance.csv")#starting with csv but can be converted to mongodb
            logging.info("Read Dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("train_test_split init")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header = True)
            logging.info("ingestion completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.init_data_ingestion()
