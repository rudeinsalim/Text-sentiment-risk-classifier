import os 
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def train_model():
    #1. Load the dataset
    data = pd.read_csv('data/sample_dataset.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}, please create it first.")
    df = pd.read_csv(data_path)
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
    print(classification_report(Y_test, predictions, zero_division=0))

    #6. Save the trained model
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join("models","classifier.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()