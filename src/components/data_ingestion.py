import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import save_object
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_path: str=os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingesion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion")
        try:
            df = pd.read_csv('notebook\data.csv')
            logging.info("Data ingestion completed")
            
            os.makedirs(os.path.dirname(self.ingesion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingesion_config.raw_data_path, index=False, header=True)
            logging.info('Initiating train test split')
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingesion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingesion_config.test_data_path, index=False, header=True)

            logging.info('Train test split completed')

            return (
                self.ingesion_config.train_data_path,
                self.ingesion_config.test_data_path
            )
        except Exception as e:
            logging.error("Error while initiating data ingestion")
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)