import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

#Initialize the FastAPI app
app = FastAPI(
    title="Text Risk & Sentiment Classifier API",
    description="This API allows you to classify text data into risk and sentiment categories using a pre-trained machine learning model.",
    version="1.0.0"
)

#Load the trained model on startup
MODEL_PATH = os.path.join("models", "classifier.joblib")
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Model not found at {MODEL_PATH}. Please train the model first.")

#Define pydantic request schema
class PredictRequest(BaseModel):
    text: str = Field(..., example="Unauthorized access attempt detected in the system logs.")

#Define pydantic response schema
class PredictResponse(BaseModel):
    text: str
    predicted: str
    confidence_score: float

@app.get("/")
def read_root():
    return {"status": "Online", "message": "Text Classification API is running."}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")

    #Predict Label
    predicted_label = model.predict([request.text])[0]

    #Extract prediction probabilities if available
    probabilities = model.predict_proba([request.text])[0] 
    confidence_score = float(max(probabilities))

    return PredictResponse(
        text=request.text,
        predicted=predicted_label,
        confidence_score=round(confidence_score, 4)
    )