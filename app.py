import uvicorn
import numpy as np
import pandas as pd

from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData,PredictPipeline
from fastapi import FastAPI, HTTPException

# Define Pydantic model for input validation
class InputData(BaseModel):
        gender: str
        race_ethnicity: str
        parental_level_of_education:str
        lunch: str
        test_preparation_course: str
        reading_score: int
        writing_score: int

# Create FastAPI app
app = FastAPI()

# Home route
@app.get("/")
def home():
    return {"message": "prediction API"}

@app.post("/predict")
def predict_datapoint(data:InputData):
    try:
        input_data=CustomData(
            gender=data.gender,
            race_ethnicity=data.race_ethnicity,
            parental_level_of_education=data.parental_level_of_education,
            lunch=data.lunch,
            test_preparation_course=data.test_preparation_course,
            reading_score=float(data.reading_score),
            writing_score=float(data.writing_score)
        )
        pred_df=input_data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return results[0]

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)