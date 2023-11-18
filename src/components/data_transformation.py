import pandas as pd 
import numpy as np 
import os 
import sys 
from dataclasses import dataclass
from src.logger import logging 
from src.exception import CustomException
from src.utils.utils import save_object 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder ,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.components.data_ingestion import DataIngestion

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformation(self):
        try:
            logging.info('Data Transformation start')
            numerical_columns = ['age', 'bmi', 'children']
            categorical_columns = ['sex', 'smoker', 'region']
            logging.info('Pipeline Started')
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer()),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoder',OneHotEncoder())
                ]
            )
            preprocessor = ColumnTransformer(
                [
                    ('categorical_columns',cat_pipeline,categorical_columns),
                    ('numerical_columns',num_pipeline,numerical_columns)
                ]
)
            logging.info('pipeline completed')
            return preprocessor

        except Exception as e:
            logging.info('error occured in the get data transformation')
            raise CustomException(e,sys)
    def initialize_data_transformation(self,train_data,test_data):
            try:
                train_df = pd.read_csv(train_data)
                test_df = pd.read_csv(test_data)
                logging.info('read train and test data')
                logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
                logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

                preprocessor_obj = self.get_data_transformation()

                target_column = 'expenses'
                drop_columns = [target_column]

                input_features_train_df  = train_df.drop(columns=target_column,axis=1)
                target_feature_train_df = train_df[target_column]

                input_features_test_df  = test_df.drop(columns=target_column,axis=1)
                target_feature_test_df = test_df[target_column]

                input_feature_train_arr = preprocessor_obj.fit_transform(input_features_train_df,target_feature_train_df)
                input_feature_test_arr = preprocessor_obj.transform(input_features_test_df)
                logging.info("Applying preprocessing object on training and testing datasets.")

                train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

                save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
                
                            )
                logging.info('data transformation is completed')
                return(
                train_arr,
                test_arr
                )
                
            except Exception as e:
                logging.info("Exception occured in the initiate_datatransformation")
                raise CustomException(e,sys)
if __name__ == "__main__":
    c =DataIngestion()
    train_data,test_data = c.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initialize_data_transformation(train_data,test_data)