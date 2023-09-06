import os
import sys
import dill
import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):

    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(x_train,x_test,y_train,y_test,models,param):
    try:
         report={}

         for i in range(len(list(models))):
             model=list(models.values())[i]
            #  para=param[list(models.key())[i]]

            #  gs=GridSearchCV(model,para)

            #  gs.fit(x_train,y_train)

            #  model.set_params(**gs.best_params_)

             model.fit(x_train,y_train)

             y_train_score=model.predict(x_train)

             y_test_score=model.predict(x_test)

             train_model_score=r2_score(y_train,y_train_score)

             test_model_score=r2_score(y_test,y_test_score)

             report[list(models.keys())[i]]=test_model_score

         return report

    except Exception as e:
        raise CustomException(e,sys)