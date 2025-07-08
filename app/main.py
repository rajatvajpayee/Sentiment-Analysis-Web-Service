from fastapi import FastAPI # The main class to create a web API.
from pydantic import BaseModel # From Pydantic; used to define the shape of input data.
from app.model import load_model

app = FastAPI()
model = load_model() #Load Once

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    prediction = model.predict([input.text])[0]
    label = "Positive" if prediction == 1 else "Negative"
    return {"prediction": label}

## To test it locally - 
# [Host] $ uvicorn app.main:app --reload
# [Host] $ curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"text": "I dont like this movie"}' 
