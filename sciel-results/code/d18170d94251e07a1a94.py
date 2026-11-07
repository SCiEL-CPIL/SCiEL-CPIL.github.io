import numpy as np

# Fixed random network weights for the basis
# Using a fixed seed ensures reproducibility and identical basis across evaluations.
# The basis is: h(x, y) = tanh(W2 @ tanh(W1 @ [x, y] + b1) + b2)
# The parameters p (length 10) are the coefficients in the final linear combination: pred = h @ p

def _init_network():
    rng = np.random.RandomState(12345)
    # First layer: 2 inputs -> 16 hidden units
    W1 = rng.randn(2, 16) * 1.0
    b1 = rng.randn(16) * 0.5
    
    # Second layer: 16 hidden -> 10 hidden (to match param length)
    W2 = rng.randn(16, 10) * 1.0
    b2 = rng.randn(10) * 0.5
    
    return W1, b1, W2, b2

# Pre-compute the network weights
W1, b1, W2, b2 = _init_network()

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y given x and y using a fixed-basis neural network model.
    The model is linear in params, allowing BFGS to converge quickly.
    
    Parameters:
    x: array of shape (N,)
    y: array of shape (N,)
    params: array of shape (10,)
    
    Returns:
    array of shape (N,)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure x and y are 1D
    x = x.flatten()
    y = y.flatten()
    
    # Input matrix of shape (N, 2)
    X = np.column_stack((x, y))
    
    # Forward pass through the fixed network
    # Layer 1: (N, 2) @ (2, 16) -> (N, 16)
    Z1 = X @ W1 + b1
    H1 = np.tanh(Z1)
    
    # Layer 2: (N, 16) @ (16, 10) -> (N, 10)
    Z2 = H1 @ W2 + b2
    H2 = np.tanh(Z2)
    
    # Final linear combination
    # H2 is (N, 10), p is (10,)
    pred = H2 @ p
    
    # Safety: replace any non-finite values with 0
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
