import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "sample_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
LABELS = ("low", "medium", "high", "critical")

def train_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}, please create it first.")
    df = pd.read_csv(DATA_PATH)
    X = df['text']
    Y = df['label']

    #2. Split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #3. Create a pipeline with TfidfVectorizer and LogisticRegression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
        ('clf', LogisticRegression(random_state=42))
    ])

    #4. Train the model
    print("Training the model...")
    pipeline.fit(X_train, Y_train)

    #5. Evaluate the model
    print("Evaluating the model...")
    predictions = pipeline.predict(X_test)
    print(classification_report(
        Y_test,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        zero_division=0,
    ))

    #6. Save the trained model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "classifier.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()