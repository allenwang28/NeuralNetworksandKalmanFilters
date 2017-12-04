import numpy as np
def MSE(actual, predictions):
    # Expect these in list form
    actual = np.array(actual)
    predictions = np.array(predictions)
    N = actual.shape[0]
    return np.linalg.norm(actual - predictions) / N
