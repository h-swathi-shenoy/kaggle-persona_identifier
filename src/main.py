from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from model import test_model
import uvicorn


app = FastAPI()


# class KaggleResponsesIn(BaseModel):
#     test_data: np.array


# class KaggleTestResponsesOut(BaseModel):
#     predictions : dict


# @app.get("/ping")
# async def pong():
#     return {"ping": "pong!"}

@app.get("/get")
async def get():
    return {"status": "OK"}


# @app.post("/predict", response_model=KaggleTestResponsesOut, status_code=200)
# def get_prediction(payload: KaggleResponsesIn):
#     test_arr = payload.test_data

#     prediction = test_model(test_arr)

#     if not prediction_list:
#         raise HTTPException(status_code=400, detail="Model not found.")

#     response_object = {"Persona": prediction}
#     return response_object

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)