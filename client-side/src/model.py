import numpy as np
import joblib
import pickle
from pathfinder import PathConfig

pathconfig = PathConfig()
models_dir = pathconfig.models_dir


def load_mapping_file() -> object:
    """
    Load the encoder file for target labels
    :return: encoder object
    """
    file = open(models_dir.joinpath("target_encoder.pkl"), "rb")
    label_obj = pickle.load(file)
    return label_obj


def test_model(test_data: np.array) -> list:
    """
    Load the saved model and return the predictions and prediction probability
    :param test_data: np.array
    :return: list
    """
    print("enter_prediction")
    model_file = models_dir.joinpath("Persona-Multiclass.pkl")
    if not model_file.exists():
        return False
    model = joblib.load(model_file)
    predictions = model.predict(test_data)
    class_probs = model.predict_proba(test_data)
    persona_prob = int(class_probs[:, predictions[0]])
    print(persona_prob)
    # if persona_prob<50:
    #     persona_prob = random.randint(60,80)
    map_obj = load_mapping_file()
    persona = map_obj.inverse_transform(predictions)
    print("Predictions", persona)
    print("Probability", persona_prob)
    return persona, persona_prob


def convert(prediction_list: list, prob_list: list) -> dict:
    """
    Convert the prediciton and prediction probablity to a dictionary
    :param prediction_list: list
    :param prob_list: list
    :return: dict
    """
    output = {}
    for i in range(len(prediction_list)):
        output[str(i)] = str(prediction_list[i])
        output['Probability'] = prob_list
    return output
