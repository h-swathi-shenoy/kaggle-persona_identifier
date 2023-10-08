from fastapi import FastAPI,Body
import numpy as np
from model import test_model, convert
import uvicorn
import json
from pathfinder import PathConfig

pathconfig = PathConfig()
app = FastAPI()


@app.get("/get")
async def get():
    return {"status": "OK"}


@app.post("/predict")
async def predictions(payload:dict = Body()) -> list:
    body = json.loads(payload['test_data'])
    test_arr = np.array(list(dict(body).values())).reshape(-1, 50)
    persona, persona_probs = test_model(test_arr)
    response = convert(persona, persona_probs)
    return [response]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)