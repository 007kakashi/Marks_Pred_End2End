import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation ,DataTransformationConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str =os.path.join('artifacts','train.csv')
    test_data_path: str =os.path.join('artifacts','test.csv')
    raw_data_path: str =os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_injegstion(self):
        logging.info('Entered the data ingestion method or components')
        try:
            data=pd.read_csv('notebook\data\stud.csv')
            logging.info('read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            data.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info('Train Test Split initiated')
            train_set,test_set=train_test_split(data,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of the data is completed')

            return(
                
                # whenver executed it will take the data from this paths

                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path,
                
            )


        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == '__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_injegstion()

    data_transformer=DataTransformation()
    train_arr,test_arr,_=data_transformer.initiate_data_trasnformer(train_data,test_data)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))


