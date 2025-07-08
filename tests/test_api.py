from fastapi.testclient import TestClient
import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import app

client = TestClient(app)

## For testcase, start the function name with test

def test_prediction():
    response = client.post("/predict", json={"text": "I love this!"})
    assert response.status_code == 200
    assert response.json()["prediction"] in ["Positive", "Negative"]

def test_Testcase2():
    response = client.post("/predict", json={"text": "I do not love this movie much, even though it had all my favourite actorz."})
    assert response.status_code == 200
    assert response.json()["prediction"] in ["Positive", "Negative"]
