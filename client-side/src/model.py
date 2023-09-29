import numpy as np
import typing as t
import random
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import pickle
from pathfinder import PathConfig

pathconfig = PathConfig()
models_dir = pathconfig.models_dir

cd
def load_mapping_file():
    file = open(models_dir.joinpath("target_encoder.pkl"), "rb")
    label_obj = pickle.load(file)
    return label_obj



def test_model(test_data: np.array) -> list:
    print("enter_prediction")
    model_file = models_dir.joinpath("Persona-Multiclass.pkl")
    if not model_file.exists():
        return False
    model = joblib.load(model_file)
    predictions = model.predict(test_data)
    class_probs = model.predict_proba(test_data)
    persona_prob = int(class_probs[:,predictions[0]])
    print(persona_prob)
    # if persona_prob<50:
    #     persona_prob = random.randint(60,80)
    map_obj = load_mapping_file()
    persona = map_obj.inverse_transform(predictions)
    print("Predictions", persona)
    print("Probability", persona_prob)
    return persona, persona_prob


def convert(prediction_list:  list, prob_list: list):
    output = {}
    for i in range(len(prediction_list)):
        output[str(i)] = str(prediction_list[i])
        output['Probability'] = prob_list
    return output

