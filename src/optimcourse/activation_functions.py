import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    res = 1 / (1 + np.exp(-x))
    return res


def relu(x: np.ndarray) -> np.ndarray:
    res = np.maximum(x, 0)
    return res


def tanh(x: np.ndarray) -> np.ndarray:
    res = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return res


def linear(x: np.ndarray) -> np.ndarray:
    res = x
    return res
