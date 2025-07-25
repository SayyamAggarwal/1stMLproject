from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.utils import load_obj,save_obj
from src.pipeline.predict_pipeline import CustomData,Predict_pipeline 
application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template("index.html")



@app.route("/predictdata",methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            writing_score=float(request.form.get('writing_score')),
            reading_score=float(request.form.get('reading_score')),
        )
        pred_df = data.get_data_as_frame()
        print(pred_df)
        predict_pipeline = Predict_pipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    
if __name__ == "__main__":
    app.run('0.0.0.0',debug=True)

    

