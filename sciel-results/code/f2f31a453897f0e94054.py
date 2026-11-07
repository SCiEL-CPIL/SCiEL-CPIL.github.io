"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Two-hidden-layer tanh neural network with 10 learnable output parameters.
    The model is linear in params, making it suitable for BFGS optimization.
    
    Architecture:
    - Input: (x, y)
    - Hidden layer 1: 16 neurons, tanh activation, fixed weights
    - Hidden layer 2: 16 neurons, tanh activation, fixed weights
    - Output: linear combination of selected h2 features + polynomial basis terms
    
    We use 10 params: 4 weights for key h2 neurons, plus 6 polynomial basis terms
    (1, x, y, x^2, y^2, x*y) to cover common functional forms.
    """
    # Fixed random weights for hidden layers (deterministic, well-scaled)
    rng = np.random.RandomState(42)
    
    n_hidden1 = 16
    n_hidden2 = 16
    
    # Hidden layer 1: input (x, y) -> 16 neurons
    # Use Xavier/Glorot initialization scaled for tanh
    W1 = rng.randn(n_hidden1, 2) * np.sqrt(1.0 / 2)
    b1 = np.zeros(n_hidden1)
    
    # Hidden layer 2: 16 neurons -> 16 neurons
    W2 = rng.randn(n_hidden2, n_hidden1) * np.sqrt(1.0 / n_hidden1)
    b2 = np.zeros(n_hidden2)
    
    # Compute hidden layer activations
    # x and y are 1D arrays of length n
    z1 = x[:, None] * W1[None, :, 0] + y[:, None] * W1[None, :, 1] + b1[None, :]
    h1 = np.tanh(z1)  # shape: (n, 16)
    
    z2 = h1 @ W2.T + b2[None, :]
    h2 = np.tanh(z2)  # shape: (n, 16)
    
    # Output: linear combination of 4 selected h2 features + 6 polynomial basis terms
    # params[0:4] -> weights for h2[:, 0], h2[:, 1], h2[:, 2], h2[:, 3]
    # params[4]  -> constant
    # params[5]  -> x
    # params[6]  -> y
    # params[7]  -> x^2
    # params[8]  -> y^2
    # params[9]  -> x*y
    
    out = (params[0] * h2[:, 0] + 
           params[1] * h2[:, 1] + 
           params[2] * h2[:, 2] + 
           params[3] * h2[:, 3] +
           params[4] +
           params[5] * x +
           params[6] * y +
           params[7] * x**2 +
           params[8] * y**2 +
           params[9] * x * y)
    
    return out
# EVOLVE-BLOCK-END
