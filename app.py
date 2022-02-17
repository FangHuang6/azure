# -*- coding: utf-8 -*-
import joblib
import json

from flask import Flask, request, jsonify
import numpy as np


with open('dummy_column_mapper.json') as fin:
    dummy_column_mapper = json.load(fin)
    
with open('scaler_info.json') as fin:
    scaler_info = json.load(fin)
    
with open('col_order.json') as fin:
    col_order = json.load(fin)
    

# make app
app = Flask(__name__)
app.config["DEBUG"] = True

# define endpoints
@app.route('/', methods=['GET'])
def home():       
            
    return 'App is Healthy'


@app.route('/predict', methods=['POST'])
def predict():
    
    payload = request.json
    for column, dummy_columns in dummy_column_mapper.items():
        for dummy_column in dummy_columns:
            payload[dummy_column] = 0
        if column in payload:
            column_val = payload.pop(column)
            target_column = f'{column}_{column_val}'
            payload[target_column] = 1

    for key, scaler_params in scaler_info.items():
        if key in payload:
            payload[key] = (payload[key] - scaler_params['mean'])/scaler_params['std']
        else:
            payload[key] = scaler_params['mean']

    ordered_payload = {}
    for col in col_order:
        ordered_payload[col] = payload[col]
    
    prob = []
    clf1 = joblib.load('svc_model.joblib')
    prediction1 = clf1.predict_proba(np.array(list(ordered_payload.values())).reshape(1, -1))[0][1]
    prob.append(prediction1)
    clf1 = joblib.load('rf_model.joblib')
    prediction2 = clf1.predict_proba(np.array(list(ordered_payload.values())).reshape(1, -1))[0][1]
    prob.append(prediction2)   
    clf1 = joblib.load('knn_model.joblib')
    prediction3 = clf1.predict_proba(np.array(list(ordered_payload.values())).reshape(1, -1))[0][1]
    prob.append(prediction3)   

    
    return prob


if __name__ == '__main__':
    app.run(debug=True)
