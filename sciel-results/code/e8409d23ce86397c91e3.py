"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function f(x, y) using a fixed single-hidden-layer neural network.
    
    Architecture:
    - Input: 2 features (x, y)
    - Hidden layer: 8 neurons with tanh activation
    - Output layer: 1 neuron (linear)
    
    Parameters (10 total):
    - params[0:8]: Weights from hidden layer to output (one per neuron)
    - params[8]: Bias for hidden layer (shared across neurons)
    - params[9]: Bias for output layer
    
    The input-to-hidden weights are fixed to a deterministic pattern to ensure
    the model has sufficient capacity while keeping the parameter count at 10.
    """
    # Ensure inputs are 1D arrays for consistent indexing
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = len(x)
    
    # Fixed input-to-hidden weights (2 x 8 matrix)
    # Using a structured pattern that covers different scales and signs
    W_in = np.array([
        [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0],
        [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.3, -0.3]
    ])
    
    # Build input matrix (n x 2)
    X = np.column_stack((x, y))
    
    # Compute hidden layer activations
    # hidden = tanh(X @ W_in + bias_hidden)
    bias_hidden = params[8]
    hidden_pre = X @ W_in + bias_hidden
    hidden = np.tanh(hidden_pre)  # (n, 8)
    
    # Compute output
    # output = hidden @ weights_out + bias_out
    weights_out = params[0:8]  # (8,)
    bias_out = params[9]
    
    pred = hidden @ weights_out + bias_out
    
    # Safety check: if any prediction is non-finite, return zeros to avoid crashing the evaluator
    if not np.all(np.isfinite(pred)):
        return np.zeros(n)
        
    return pred
# EVOLVE-BLOCK-END
