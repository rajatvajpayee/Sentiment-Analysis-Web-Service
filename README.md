# Sentiment Analysis REST API with FastAPI & Docker

This project builds, trains, and deploys a sentiment analysis model on the IMDb dataset using FastAPI and Docker. It exposes a REST API for making predictions and is ready for integration into CI/CD workflows and cloud deployments.

## Components Overview

| Component        |  Purpose                                                             |
|------------------|----------------------------------------------------------------------|
| Dataset	         |  IMDb movie reviews (binary sentiment classification)                | 
| ML Pipeline	     |  Data preprocessing + Logistic Regression using scikit-learn         |       
| REST API	       |  FastAPI to serve the model and accept POST requests                 |
| Containerization |  Docker to encapsulate dependencies and deployment                   |         
| Model Versioning |  MLflow to log models, parameters, and metrics per run               | 
| CI/CD	           |  GitHub Actions to automate training, testing, and deployment steps  |               

## Project Structure 
```
Sentiment-Analysis-Web-Service/
├── .github/workflows/
│   └── ci-cd.yml                 # GitHub Actions pipeline
├── app/
│   ├── main.py                   # FastAPI app
│   ├── model.py                  # Model loader
├── train/
│   └── train_model.py            # Training + MLflow logging
├── tests/
│   └── test_api.py               # Simple unit test
├── Dockerfile
├── requirements.txt
├── mlruns/                       # MLflow artifacts (auto-generated)
├── README.md
```

## Dataset
- Source: IMDb Large Movie Review Dataset
- Accessed via: HuggingFace datasets library
- Size: 25,000 labeled movie reviews for training

## Installation 
#### Clone the repo 
```
git clone https://github.com/rajatvajpayee/Sentiment-Analysis-Web-Service.git
cd Sentiment-Analysis-Web-Service
```

#### Install Python Dependencies 
```
[Sentiment-Analysis-Web-Service] $ pip install -r requirements.txt
```

## Train the Model 
The model is trained using a Logistic Regression classifier on TF-IDF features. Preprocessing includes token cleaning, punctuation removal, and stop word filtering.
To train - 
```
[Sentiment-Analysis-Web-Service] $ python train/train_model.py
```

This will:
- Download the IMDb dataset
- Preprocess text
- Train the model
- Save the model to app/sentiment_model.joblib

## View Training Runs in MLflow UI
MLflow tracks all model training runs and logs them under the `mlruns/` directory.

To visualize training results: `mlflow ui`

Then open your browser and go to: http://127.0.0.1:5000

From the MLflow dashboard, you can:
- Browse all experiment runs
- Compare accuracy metrics
- Inspect hyperparameters
- Download trained models
- View logged artifacts

This is useful for model versioning and comparison in an MLOps workflow.

## Running the API Locally

### Start with Uvicorn (Dev Mode)
```
[Sentiment-Analysis-Web-Service] $ uvicorn app.main:app --reload
```

### Test the Endpoint
**Swagger UI** : http://127.0.0.1:8000/docs

**Sample Request** :  
```
[Sentiment-Analysis-Web-Service] $ curl -X POST http://127.0.0.1:8000/predict \
                                            -H "Content-Type: application/json" \
                                            -d '{"text": "The movie was excellent!"}'
```

**Response** : 
```
{"prediction": "positive"}
```
##  Run Unit Tests with Pytest
This project includes a test suite located in the `tests/` directory. These tests ensure that the REST API is working as expected and returns valid sentiment predictions.

###  To run the tests:
```
[Sentiment-Analysis-Web-Service] $ pytest

============================= test session starts =============================
collected 1 item

tests/test_api.py .                                                   [100%]

============================== 1 passed in 0.45s =============================
```

Make sure the model is already trained (app/sentiment_model.joblib exists), or run: `python train/train_model.py`

## Docker Deployment
```
## Build the image
[Sentiment-Analysis-Web-Service] $ docker build -t sentiment-service . 

## Run the container
[Sentiment-Analysis-Web-Service] $ docker run -p 8000:8000 sentiment-service

## Access the API 
[Sentiment-Analysis-Web-Service] $  curl -X POST http://localhost:8000/predict \
                                          -H "Content-Type: application/json" \
                                          -d '{"text": "This was a terrible experience"}'
```

## API Reference

**Request Body** :
```
{
  "text": "string"
}
```

**Response** :
```
{
  "prediction": "positive" | "negative"
}
```

## Requirements
- Python 3.8+
- fastapi, scikit-learn, joblib, nltk, datasets, uvicorn

```
[Sentiment-Analysis-Web-Service] $ pip install -r requirements.txt
```

## Future Improvements 
- Add confidence score in predictions
- Support batch predictions
- Integrate MLflow for tracking
- Store models in S3 or GCS
