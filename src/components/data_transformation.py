import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        
        '''
        this function for data transformer

        '''

        logging.info('Entered into transformer')

        try:
            numerical_feature=[
                'reading_score', 
                'writing_score'
                ]

            categorical_feature=[
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch',       
                'test_preparation_course'
                ]
            
            numerical_pipeline=Pipeline(
                steps=[
                     ('Imputer',SimpleImputer(strategy='median')),
                     ('Scaler',StandardScaler())
                ]
            )

            logging.info('Numerical scaling completed')

            categorical_pipeline=Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='most_frequent')),
                    ('One Hot Encoder',OneHotEncoder()),
                    ('Scaler',StandardScaler(with_mean=False))
                ]
            )
            
            logging.info('Categorical encoding completed')
            

            preprocessor=ColumnTransformer(
                [
                    ('Num_pipeline',numerical_pipeline,numerical_feature),
                    ('Cat_pipeline',categorical_pipeline,categorical_feature)
                ]
            )    
            
            logging.info("Enterd in preprocessor")

            return preprocessor

            


        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_trasnformer(self,train_path,test_path):

        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            logging.info('Read train adn test data completed')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformer_obj()

            target_column_name='math_score'
            
            # numerical_column=self.numerical_feature
            # categorical_column= self.categorical_feature

            input_feature_train_data=train_data.drop(target_column_name,axis=1)
            target_feature_train_data=train_data[target_column_name]

            input_feature_test_data=test_data.drop(target_column_name,axis=1)
            target_feature_test_data=test_data[target_column_name]

            logging.info('Applying preprocessor object to train dataframe and testing dataframe.')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_data)
    

            train_arr=np.c_[

                input_feature_train_arr,np.array(target_feature_train_data)

            ]

            test_arr=np.c_[

                input_feature_test_arr,np.array(target_feature_test_data)


            ]

            logging.info('Transformation Done Succesfuly')

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(

                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )



        except Exception as e:
            raise CustomException(e,sys)
            