from flask import Flask,request,render_template
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.rount('/predictdata',method=['GET','POST'])
def predict_datapoints():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData()