import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from pathlib import Path
from utils.metrics_report import get_multiclass_report

current_dir = Path.cwd()
processed_data_dir = current_dir.joinpath('data/processed')
train_dir =  processed_data_dir.joinpath('train-data')
test_dir = processed_data_dir.joinpath('test-data')
models_dir = current_dir.joinpath('models/artifacts')


def train_model(train_data: pd.DataFrame) -> None:
    """
    Train a Random Forest Classifier model with hyperparams and save it.
    Parameters
    ----------
    train_data: pd.DataFrame
    """
    X = train_data.iloc[:, :-1].values
    y = train_data.iloc[:, -1:].to_numpy().ravel()
    params =  {'n_estimators': 463, 'max_depth': 50}
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X, y)
    joblib.dump(model, models_dir.joinpath("Persona-Multiclass.joblib"))


def test_model(test_data: pd.DataFrame) -> np.ndarray:
    """
    Load the saved model and predict on test set
    Parameters
    ----------
    test_data :pd.DataFrame

    Returns:array
    -------

    """
    model_file = models_dir.joinpath("Persona-Multiclass.joblib")
    if not model_file.exists():
        return False
    model = joblib.load(model_file) 
    actual_test = test_data.drop(columns = ['Target'])
    test_arr = actual_test.values
    predictions = model.predict(test_arr)
    return predictions


if __name__ == "__main__" :
    train_data = pd.read_csv(train_dir.joinpath('KaggleResponses-Train.csv'))
    test_data =  pd.read_csv(test_dir.joinpath('KaggleResponses-Test.csv'))
    y_test = test_data['Target']
    # train_model(train_data)
    # print("Model saved")
    predictions = test_model(test_data)
    get_multiclass_report("RF Classification", "Chisquare",predictions, y_test)
    print("Test predictions done")