import os
import string
import joblib
import mlflow
import mlflow.sklearn
import nltk
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import re
from datasets import load_dataset
from nltk.corpus import stopwords

import uuid
model_name = f"sentiment_model_{uuid.uuid4().hex[:8]}"

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess(text):
    """
    Preprocesses the input text by applying the following steps:
    1. Converts all characters to lowercase.
    2. Removes all punctuation characters.
    3. Removes all numeric digits.
    4. Removes common stop words.
    
    Parameters:
        text (str): The input text string to be cleaned.
    
    Returns:
        str: The cleaned and preprocessed text.
    
    Note:
    """
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def main():
    print("Loading dataset...")
    dataset = load_dataset("imdb", split="train")
    texts = [preprocess(example["text"]) for example in dataset]
    labels = dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

    print("Training model...")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))

    print("Saving model...")
    os.makedirs("app", exist_ok=True)
    joblib.dump(model, "app/sentiment_model.joblib")

    # Log to MLflow
    mlflow.set_experiment("SentimentAnalysis")
    with mlflow.start_run():
        mlflow.log_params({
            "model": "LogisticRegression",
            "vectorizer": "TfidfVectorizer",
            "stop_words": True,
            "max_features": 10000
        })
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, model_name)
        mlflow.log_artifact("train/train_model.py")

        print("Training complete. Model logged to MLflow.")

if __name__ == "__main__":
    main()