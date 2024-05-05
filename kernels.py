import numpy as np

def GRBF(x1, x2):
    diff = x1 - x2
    return np.exp(-np.dot(diff, diff) / 2)