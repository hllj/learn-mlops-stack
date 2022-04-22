from fastapi import FastAPI
from inference_onnx import ColaONNXPredictor


app = FastAPI(title='MLOPs Basics App')
predictor = ColaONNXPredictor("./models/model.onnx")

@app.get("/")
async def home():
    return "Hello World"

@app.get("/predict")
async def get_predictions(text: str):
    result = predictor.predict(text)
    return {
        'result': result
    }
