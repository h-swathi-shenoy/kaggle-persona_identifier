from sklearn.metrics import classification_report
import numpy as np


def get_multiclass_report(
    model_name: str, feat_selection: str, y_preds: np.array, y_actual: np.array
) -> classification_report:
    metrics = classification_report(y_actual, y_preds)
    print(f"Metrics for {model_name} & {feat_selection} are \n {metrics}")
    return
