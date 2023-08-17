import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):

        '''This function will return the data transformer object'''

        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course",]
            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler())
            ])
            logging.info("Numerical columns standard scaling Completed")
            logging.info("Categorical Encoding Completed")

            preprocessor = ColumnTransformer(
                [
                    ('numerical', numerical_pipeline, numerical_columns),
                    ('categorical', categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)    
    
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Data ingestion completed")
            
            # os.makedirs(os.path.dirname(self.transformation_config.preprocessor_ob_file_path), exist_ok=True)

            preprocessor_ob = self.get_data_transformer_object()
            logging.info("Data transformation completed")

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on train and test dataframe")

            input_feature_train_arr = preprocessor_ob.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_ob.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saving preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessor_ob
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_ob_file_path
            )
        except Exception as e:
            logging.error("Error while initiating data transformation")
            raise CustomException(e,sys)
        