from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import pandas as pd
import pickle

from db import get_db, create_table
from sqlalchemy.orm import Session
from schemas import UserInput
from services import save_user_input


# Load ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


app = FastAPI()

# create table if not exists
create_table()


# ---------- Frontend Setup ----------
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------- Prediction API ----------
@app.post("/predict")
def predict_premium(data: UserInput, db: Session = Depends(get_db)):

    # Name validation
    if not data.name.strip():
        raise HTTPException(status_code=400, detail="Name is required!")

    # Prepare ML input
    input_df = pd.DataFrame([{
        "gender": data.genderr,
        "ap_hi": data.ap_hi,
        "ap_lo": data.ap_lo,
        "cholesterol": data.cholesterol,
        "gluc": data.gluc,
        "smoke": data.smoke,
        "alco": data.alco,
        "active": data.active,
        "BMI_Category": data.bmi_category,
        "age_group": data.age_group
    }])

    # Prediction
    prediction = int(model.predict(input_df)[0])

    # Save user input + prediction
    save_user_input(db, data, prediction=prediction, name=data.name.strip())

    # Return result
    return JSONResponse(
        status_code=200,
        content={"predicted_category": prediction}
    )