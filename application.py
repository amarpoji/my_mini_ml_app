from fastapi import FastAPI
import pickle
import os

application = FastAPI()   # MUST be named application for EB

# load model safely
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# routes
@application.get("/")
def home():
    return {"message": "House Price Predictor API is running"}

@application.get("/predict")
def predict(size: float):
    price = model.predict([[size]])[0]
    return {"size": size, "predicted_price": price}

@application.get("/health")
def health():
    return {"status": "ok"}
