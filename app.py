from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    
    try:
        # Extract form data with validation
        gender = request.form.get('gender')
        ethnicity = request.form.get('ethnicity')
        parental_education = request.form.get('parental_level_of_education')
        lunch = request.form.get('lunch')
        test_prep = request.form.get('test_preparation_course')
        reading_score = request.form.get('reading_score')
        writing_score = request.form.get('writing_score')
        
        # Validate required fields
        if not all([gender, ethnicity, parental_education, lunch, test_prep, reading_score, writing_score]):
            return render_template('index.html', error="Please fill in all fields.")
        
        # Convert scores to float
        try:
            reading_score = float(reading_score)
            writing_score = float(writing_score)
        except ValueError:
            return render_template('index.html', error="Please enter valid numeric values for scores.")
        
        # Validate score ranges
        if not (0 <= reading_score <= 100) or not (0 <= writing_score <= 100):
            return render_template('index.html', error="Scores must be between 0 and 100.")
        
        # Create CustomData object
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_education,
            lunch=lunch,
            test_preparation_course=test_prep,
            reading_score=reading_score,
            writing_score=writing_score
        )
        
        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        print("Data DataFrame:")
        print(pred_df)
        
        # Make prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Round the result to 2 decimal places
        predicted_score = round(float(results[0]), 2)
        
        # Ensure prediction is within reasonable bounds
        predicted_score = max(0, min(100, predicted_score))
        
        return render_template('index.html', results=predicted_score)
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"Error during prediction: {str(e)}")
        return render_template('index.html', error=error_msg)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
