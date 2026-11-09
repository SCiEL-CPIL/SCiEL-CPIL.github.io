"""Symbolic regression with 2 inputs: 2-hidden-neuron tanh network with linear output."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    2-hidden-layer tanh neural network with 10 parameters.
    
    params = [w11, w12, b1,  w21, w22, b2,  v1, v2, v3, b3]
    
    h1 = tanh(w11*x + w12*y + b1)
    h2 = tanh(w21*x + w22*y + b2)
    pred = v1*h1 + v2*h2 + v3*x + b3
    """
    if time.time() > _DEADLINE:
        # Fallback: simple linear model
        return params[0] * x + params[1] * y + params[2]

    w11, w12, b1, w21, w22, b2, v1, v2, v3, b3 = params

    h1 = np.tanh(w11 * x + w12 * y + b1)
    h2 = np.tanh(w21 * x + w22 * y + b2)
    pred = v1 * h1 + v2 * h2 + v3 * x + b3

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        mask = np.isfinite(pred)
        pred[~mask] = 0.0

    return pred
# EVOLVE-BLOCK-END
