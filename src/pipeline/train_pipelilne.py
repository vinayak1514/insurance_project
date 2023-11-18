from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.logger import logging 
from src.exception import CustomException

data_ingestion =DataIngestion()
train_data,test_data = data_ingestion.initiate_data_ingestion()

data_transformation = DataTransformation()
train_array,test_array = data_transformation.initialize_data_transformation(train_data,test_data)

model_train = ModelTrainer()
model_train.initiate_model_trainer(train_array,test_array)