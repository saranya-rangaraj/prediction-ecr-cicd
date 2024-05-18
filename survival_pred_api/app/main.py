import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import gradio
from fastapi import FastAPI, Request, Response

import random
import numpy as np
import pandas as pd
import joblib


# FastAPI object
app = FastAPI()


# UI - Input components
user_inputs = [
                gradio.Slider(5, 120),
                gradio.Radio([0, 1]),
                gradio.Slider(20, 8000),
                gradio.Radio([0, 1]),
                gradio.Slider(10, 80),
                gradio.Radio([0, 1]),
                gradio.Slider(25100, 850000),
                gradio.Slider(0.5, 10),
                gradio.Slider(110, 150),
                gradio.Radio([0, 1]),
                gradio.Radio([0, 1]),
                gradio.Slider(1, 300),
              ]

# UI - Output component
out_label = gradio.Textbox(type="text", label='Prediction', elem_id="out_textbox")

# load trained model
save_file_name = "xgboost-model.pkl"
trained_model = joblib.load(filename=save_file_name)

# Label prediction function
def predict_death_event(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time):
    input_data = {
        "age" : age,
        "anaemia" : anaemia,
        "creatinine_phosphokinase": creatinine_phosphokinase,
        "diabetes": diabetes,
        "ejection_fraction": ejection_fraction,
        "high_blood_pressure": high_blood_pressure,
        "platelets": platelets,
        "serum_creatinine": serum_creatinine,
        "serus_sodium": serum_sodium,
        "sex": sex,
        "smoking": smoking,
        "time": time,
    }
    input_df = pd.DataFrame([input_data])
    if trained_model.predict(input_df)[0] == 1:
        return "patient deceased"
    else:
        return "patient survived"

# Create Gradio interface object
title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gradio.Interface(fn = predict_death_event,
                         inputs = user_inputs,
                         outputs = out_label,
                         title = title,
                         description = description,
                         allow_flagging='never')

# Mount gradio interface object on FastAPI app at endpoint = '/'
app = gradio.mount_gradio_app(app, iface, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
