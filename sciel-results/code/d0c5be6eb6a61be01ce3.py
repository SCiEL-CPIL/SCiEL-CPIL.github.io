"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Single hidden layer neural network with 10 parameters.
    Architecture: x -> [3 tanh units] -> 1 output
    params[0:3] = weights from input to hidden layer
    params[3:6] = biases of hidden layer
    params[6:9] = weights from hidden layer to output
    params[9]   = output bias
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Hidden layer: 3 tanh units
    # h_j = tanh(params[0]*x + params[3], params[1]*x + params[4], params[2]*x + params[5])
    h0 = np.tanh(params[0] * x + params[3])
    h1 = np.tanh(params[1] * x + params[4])
    h2 = np.tanh(params[2] * x + params[5])
    
    # Output layer
    pred = params[6] * h0 + params[7] * h1 + params[8] * h2 + params[9]
    
    return pred
# EVOLVE-BLOCK-END
