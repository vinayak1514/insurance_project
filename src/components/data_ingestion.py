import os
import sys
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from pathlib import Path
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

class DataIngestionConfig:
    raw_data_path:str = os.path.join('artifacts','raw.csv')
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('data ingestion started')
        try:
            data = pd.read_csv('D:\\study\Data_science\\code\machine_learning\\insurance_project\\notebook\\insurance.csv')
            logging.info('read Data as DataFrame')
            

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            logging.info('Save the raw data in the artifacts folder')

            train_data,test_data = train_test_split(data,test_size=0.2)
            logging.info('train test split completed')

            train_data.to_csv(self.data_ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.data_ingestion_config.test_data_path,index=False)
            logging.info('Data ingestion complete')

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info('error in the ingestion part')
            raise CustomException(e,sys)