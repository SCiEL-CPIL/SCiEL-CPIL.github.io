"""Symbolic regression with 2 inputs using a nonlinear feedforward network."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions using a nonlinear feedforward network.
    
    Architecture:
    - Input: (x, y)
    - Hidden layer 1: 5 tanh units with fixed weights
    - Hidden layer 2: 5 tanh units with fixed weights  
    - Output: linear combination of 10 features (bias + 9 hidden outputs)
    
    The 10 parameters are the output layer weights that BFGS optimizes.
    The fixed internal weights create nonlinear basis functions.
    """
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    
    # Extract parameters
    p = params[:10]
    
    # Fixed random weights for hidden layers (deterministic, not learned)
    # Hidden layer 1: 5 units, each takes (x, y) as input
    # Using fixed pseudo-random weights based on index
    np.random.seed(42)
    W1 = np.random.randn(5, 2) * 2.0  # Scale to encourage diverse activations
    b1 = np.random.randn(5) * 0.5
    
    # Hidden layer 2: 5 units, each takes 5 inputs from h1
    W2 = np.random.randn(5, 5) * 2.0
    b2 = np.random.randn(5) * 0.5
    
    # Compute hidden layer 1: h1 = tanh(x*W1[0] + y*W1[1] + b1)
    h1_input = x[:, np.newaxis] * W1[:, 0] + y[:, np.newaxis] * W1[:, 1] + b1[np.newaxis, :]
    h1 = np.tanh(h1_input)  # Shape: (n, 5)
    
    # Compute hidden layer 2: h2 = tanh(h1 @ W2 + b2)
    h2_input = h1 @ W2 + b2[np.newaxis, :]
    h2 = np.tanh(h2_input)  # Shape: (n, 5)
    
    # Output: linear combination of bias + h1 + h2 (total 1 + 5 + 5 = 11, but we only have 10 params)
    # Use bias + all h1 (5) + first 4 of h2 = 10 parameters
    # Or: bias + 4 from h1 + 5 from h2 = 10 parameters
    # Let's use: p[0] as bias, p[1:6] weight h1, p[6:10] weight first 4 of h2
    
    pred = (p[0] + 
            np.sum(h1 * p[1:6][np.newaxis, :], axis=1) +
            np.sum(h2[:, :4] * p[6:10][np.newaxis, :], axis=1))
    
    return pred
# EVOLVE-BLOCK-END
