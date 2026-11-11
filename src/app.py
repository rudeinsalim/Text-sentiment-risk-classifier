import os
import joblib
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "classifier.joblib"
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Model not found at {MODEL_PATH}. Please train the model first.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

#Initialize the FastAPI app
app = FastAPI(
    title="Text Risk & Sentiment Classifier API",
    description="This API allows you to classify text data into risk and sentiment categories using a pre-trained machine learning model.",
    version="1.0.0",
    lifespan=lifespan
)

#Define pydantic request schema
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        json_schema_extra={
            "example": "Unauthorized access attempt detected in the system logs."
        }
    )

#Define pydantic response schema
class PredictResponse(BaseModel):
    text: str
    predicted: str
    confidence_score: float

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Text Risk Classifier</title>
        <style>
          :root { color-scheme: light; font-family: system-ui, sans-serif; }
          body { margin: 0; min-height: 100vh; display: grid; place-items: center;
                 background: #f3f6fb; color: #172033; }
          main { width: min(640px, calc(100% - 32px)); padding: 32px;
                 box-sizing: border-box; background: white; border-radius: 16px;
                 box-shadow: 0 12px 40px #17203318; }
          h1 { margin-top: 0; }
          p { color: #526078; line-height: 1.5; }
          label { display: block; margin: 24px 0 8px; font-weight: 600; }
          textarea { width: 100%; min-height: 150px; padding: 12px;
                     box-sizing: border-box; border: 1px solid #cbd3df;
                     border-radius: 8px; font: inherit; resize: vertical; }
          button { margin-top: 14px; padding: 11px 18px; border: 0;
                   border-radius: 8px; background: #2459c4; color: white;
                   font: inherit; font-weight: 600; cursor: pointer; }
          button:disabled { opacity: .65; cursor: wait; }
          #result { margin-top: 20px; padding: 16px; border-radius: 8px;
                    background: #f3f6fb; }
          #result[hidden] { display: none; }
          #error { color: #b42318; }
          .note { margin-top: 20px; font-size: .9rem; }
        </style>
      </head>
      <body>
        <main>
          <h1>Text Risk Classifier</h1>
          <p>Enter text to classify its risk level as low, medium, high, or critical.</p>
          <form id="predict-form">
            <label for="text">Text to analyze</label>
            <textarea id="text" name="text" required
                      placeholder="Paste text, an alert, or a log message..."></textarea>
            <button id="submit" type="submit">Analyze text</button>
          </form>
          <section id="result" aria-live="polite" hidden>
            <strong>Predicted risk level:</strong> <span id="prediction"></span>
            <br>
            <strong>Confidence:</strong> <span id="confidence"></span>
          </section>
          <p id="error" role="alert"></p>
          <p class="note">This model predicts risk categories only. It does not currently
            produce a separate sentiment prediction.</p>
        </main>
        <script>
          const form = document.getElementById("predict-form");
          const button = document.getElementById("submit");
          const result = document.getElementById("result");
          const error = document.getElementById("error");

          form.addEventListener("submit", async (event) => {
            event.preventDefault();
            button.disabled = true;
            error.textContent = "";
            result.hidden = true;

            try {
              const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: document.getElementById("text").value })
              });
              const data = await response.json();
              if (!response.ok) {
                throw new Error(data.detail || "Prediction failed.");
              }
              document.getElementById("prediction").textContent = data.predicted;
              document.getElementById("confidence").textContent =
                `${(data.confidence_score * 100).toFixed(1)}%`;
              result.hidden = false;
            } catch (err) {
              error.textContent = err.message || "Unable to reach the prediction service.";
            } finally {
              button.disabled = false;
            }
          });
        </script>
      </body>
    </html>
    """

@app.get("/health")
def health_check():
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