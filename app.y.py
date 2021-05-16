import numpy as np
import pandas as pd
from flask import Flask,request, jsonify, render_template
import pickle


app=Flask(__name__)
model=pickle.load(open('diabetes-prediction-Logistic-model.pkl','rb'))

dataset=pd.read_csv('C:/Users/asus/Desktop/Diabetes/diabetes.csv')
dataset.head()

dataset_X = dataset.iloc[:,[1, 2, 5, 7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)



@app.route('/')

def home():
    return render_template('Index1.html')

@app.route("/predict", methods=['POST'])
def predict():
    
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( sc.transform(final_features) )

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template('Index1.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)