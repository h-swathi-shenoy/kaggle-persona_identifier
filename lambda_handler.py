import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import os
from src.model import test_model, convert
from src.pathfinder import PathConfig

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

path_obj = PathConfig()
base_path = path_obj.base_path
test_folder = path_obj.test_dir

def handler(event, context):
    """
    Lambda function handler for predicting persona of the given user inputs for questionnaire
    :param event: post/get request context
    :param context: Context of Meta data
    """
    logger.info("========> Started.")
    logger.info(f"Event type: {type(event)}")
    if event and "test_data" not in event:
        logger.info("Event found!")
        logger.info(event)
        logger.info("Seriallizing the data!")
        test_arr = np.array(list(event.values())).reshape(-1,50)
        logger.info("Predicting the data!")
        persona, persona_probs = test_model(test_arr)
        logger.info("De-serializing the response")
        response = convert(persona,persona_probs)
        logger.info("Finished. <=========")
        return {"statusCode": 200, "result": json.dumps(response)}

    if "test_data" in event:
        logger.info("Resource not found!")
        body = json.loads(event['test_data'])
        test_arr = np.array(list(dict(body).values())).reshape(-1, 50)
        persona, persona_probs = test_model(test_arr)
        response = convert(persona, persona_probs)
        return response

if __name__ == "__main__":
    test_data = pd.read_csv(test_folder.joinpath('Kaggle-Responses_v2.csv'))
    test_sample = test_data.iloc[:,:-1].values[15]
    json_data = json.dumps(dict(enumerate(test_sample.flatten(), 1)))
    json_dict= {"test_data": json_data}
    test = json_dict
    result = handler(test, None)
    print(result)