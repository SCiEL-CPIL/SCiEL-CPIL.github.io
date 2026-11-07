import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation for symbolic regression with 2 inputs x, y.
    Uses a single fully differentiable neural network with two hidden layers
    (tanh activations) where the 10 parameters are the weights of the final
    output layer. The architecture is linear in the optimizable variables
    for BFGS while providing universal approximation capability through
    fixed internal non-linear transformations.
    """
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Flatten inputs to 1D for consistent processing
    n = x.shape[0]
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    
    # Input vector: [x, y, 1] for each point
    # Shape: (n, 3)
    X = np.column_stack([x_flat, y_flat, np.ones(n)])
    
    # First hidden layer: 8 neurons, tanh activation
    # Fixed weights (not optimized)
    W1 = np.array([
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [0.5, 0.5, 1.0],
        [0.5, -0.5, 1.0],
        [-0.5, 0.5, 1.0],
        [-0.5, -0.5, 1.0]
    ])  # Shape: (8, 3)
    
    b1 = np.zeros(8)
    
    # Hidden layer 1: (n, 3) @ (8, 3).T -> (n, 8), add bias, tanh
    h1 = np.tanh(X @ W1.T + b1)  # Shape: (n, 8)
    
    # Second hidden layer: 8 neurons, tanh activation
    # Fixed weights
    W2 = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ])  # Shape: (8, 8)
    
    b2 = np.zeros(8)
    
    # Hidden layer 2: (n, 8) @ (8, 8).T -> (n, 8), add bias, tanh
    h2 = np.tanh(h1 @ W2.T + b2)  # Shape: (n, 8)
    
    # Output layer: 1 neuron, linear activation
    # The 10 parameters are used here. Since h2 has 8 features,
    # we need to map 10 params to produce a single output.
    # We can use the first 8 params as weights for h2, and the last 2
    # as additional bias-like terms or for a small extension.
    
    # Actually, let's reconsider: we have 10 params but only 8 hidden units.
    # Let's use 8 hidden units in the second layer, and then the output
    # is a linear combination of h2 (8 values) plus 2 extra basis terms
    # derived from the inputs to use all 10 params.
    
    # Alternative: Make the second hidden layer have 10 neurons so that
    # the 10 params are exactly the output weights.
    
    # Let's redo with 10 hidden neurons in the second layer.
    
    # Second hidden layer: 10 neurons, tanh activation
    W2_10 = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3]
    ])  # Shape: (10, 8)
    
    b2_10 = np.zeros(10)
    
    # Recompute hidden layer 2 with 10 neurons
    h2_10 = np.tanh(h1 @ W2_10.T + b2_10)  # Shape: (n, 10)
    
    # Output: linear combination of h2_10 with the 10 params
    pred = h2_10 @ p  # Shape: (n,)
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
