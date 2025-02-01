from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

model1 = joblib.load('model1.pkl')
model2 = joblib.load('model2.pkl')
model3 = joblib.load('model3.pkl')


scaler = joblib.load("scaler.pkl")  


def preprocess_input(data):  #Defining a Function to Preprocess the Data
    try:
        feature_names = [
            "CreditScore", "Gender", "Age", "Tenure","Balance", 
            "NumOfProducts","HasCrCard", "IsActiveMember", "EstimatedSalary",
            "Geography_France", "Geography_Germany", "Geography_Spain"
        ]
        to_scale = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']
        

        input_df = pd.DataFrame([data], columns=feature_names)


        input_df[to_scale] = scaler.transform(input_df[to_scale])  
        scaled_data = input_df
        print("Preprocessed :", scaled_data) 
        return scaled_data
    except Exception as e:
        raise ValueError(f"Preprocessing Error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Preprocessing Error: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():      #Defining a Function to Predict Churn
    try:
        content = request.get_json()
        print("Received JSON:", content)  
        
        if 'data' not in content:
            return jsonify({"error": "Missing 'data' key in request"}), 400

        input_data = content['data']
        print(input_data)

      
        processed_data = preprocess_input(input_data)

        
        pred1 = model1.predict(processed_data)
        pred2 = model2.predict(processed_data)
        pred3 = model3.predict(processed_data)

        
        final_prediction = int((pred1 + pred2 + pred3).sum() > 1)

        return jsonify({"final_prediction": final_prediction})
    
    except Exception as e:
        print("Error:", str(e))  
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
