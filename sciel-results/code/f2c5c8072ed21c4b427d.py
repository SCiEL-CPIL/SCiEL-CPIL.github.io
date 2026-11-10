import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Gaussian RBF model with 5 fixed centers.
    First 5 params control weights, last 5 control Gaussian widths.
    """
    # Normalize input to [-1, 1]
    t = x / 3.0
    
    # Five fixed, evenly spaced centers across [-1, 1]
    centers = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # Weights: first 5 params
    weights = params[0:5]
    
    # Widths: last 5 params, using shifted exponential for positive widths
    # exp(p - 1) gives width ~1 when p=1 (initial value), range (0, inf)
    widths = np.exp(params[5:10] - 1.0)
    
    # Compute RBF features
    # For each center c_i and width w_i: exp(-(t - c_i)^2 / (2 * w_i^2))
    y = np.zeros_like(t)
    for i in range(5):
        y += weights[i] * np.exp(-((t - centers[i]) ** 2) / (2.0 * widths[i] ** 2))
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
    
    return y
