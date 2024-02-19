import pickle
#import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from flask import Flask,request,jsonify,render_template 


app = Flask(__name__)
#app = application

## import ridge regressor and standard scaler pickle
classifier_model = pickle.load(open('models/ModelForprediction.pkl','rb'))
standard_scaler = pickle.load(open('models/standardScaler.pkl','rb'))

@app.route("/")
def index():
    #return "hi"
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET','POST'])
def predict_data():
    if request.method=='POST':
        Pregnancies=float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))
        
        new_data_scaled=standard_scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=classifier_model.predict(new_data_scaled)

        if predict[0]==1:
            result='Diabetic'
        else:
            result='None-Diabetic'
        return render_template('single_prediction.html',result=result)
    else:
        return render_template("home.html")



if __name__=='main':
    app.run(host='0.0.0.0',port=8080)