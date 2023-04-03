import numpy as np


def continuous_to_binary(y_array: np.array, threshold: float = None) -> list:
    """Convert an np.array with continuous values to boolean according to the threshold specified. The threshold is defined according to the **define_best_threshold** method.

    Args:
        y_array (np.array): np.array with continuous values

        threshold (str): default threshold

    Returns:
        list: list of boolean values
    """

    if threshold is None:
        threshold = 0.5
    y_array = [1 if i > threshold else 0 for i in y_array]

    return y_array
