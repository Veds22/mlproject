from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

import sys

app = Flask(__name__)

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else: 
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score')),
            )
            pred_df = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('home.html', results=results[0])
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    app.run(debug=True)