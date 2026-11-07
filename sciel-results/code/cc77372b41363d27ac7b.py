import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear feedforward network with two hidden layers (tanh activations).
    
    Architecture:
      Input: [x, y]
      Hidden layer 1: 4 neurons with tanh activation
      Hidden layer 2: 4 neurons with tanh activation
      Output: linear combination of [1, h2_1, h2_2, h2_3, h2_4, x, y, x^2, y^2, x*y]
    
    The 10 parameters are the output weights. The internal weights are fixed
    using a deterministic pseudo-random initialization based on a simple LCG,
    ensuring reproducibility and avoiding any external randomness.
    
    This allows the model to learn complex, non-separable functional forms
    through the fixed internal nonlinearity.
    """
    # Fixed internal weights using a deterministic LCG
    # Seed-based pseudo-random initialization
    def _lcg(seed, n):
        """Generate n pseudo-random numbers in [-1, 1] using a simple LCG."""
        state = seed
        result = []
        for _ in range(n):
            state = (state * 1103515245 + 12345) % (2**31)
            result.append((state / (2**30) - 1.0) * 0.5)
        return np.array(result)
    
    # Initialize hidden layer weights deterministically
    # Layer 1: 2 inputs -> 4 hidden units
    W1 = np.array([
        [-0.8, 0.5, 0.9, -0.3],
        [0.7, -0.6, 0.4, 0.8]
    ])
    b1 = np.array([0.1, -0.2, 0.3, -0.1])
    
    # Layer 2: 4 hidden units -> 4 hidden units
    W2 = np.array([
        [0.5, -0.7, 0.3, 0.9],
        [-0.4, 0.6, -0.8, 0.2],
        [0.9, 0.1, -0.5, -0.6],
        [-0.2, 0.8, 0.7, -0.4]
    ])
    b2 = np.array([-0.3, 0.2, -0.1, 0.4])
    
    # Forward pass
    # Input: [x, y]
    X = np.column_stack((x, y))  # shape: (n, 2)
    
    # Hidden layer 1
    h1 = np.tanh(X @ W1 + b1)  # shape: (n, 4)
    
    # Hidden layer 2
    h2 = np.tanh(h1 @ W2 + b2)  # shape: (n, 4)
    
    # Construct output features: [1, h2_1, h2_2, h2_3, h2_4, x, y, x^2, y^2, x*y]
    ones = np.ones_like(x)
    out_features = np.column_stack((
        ones,
        h2[:, 0], h2[:, 1], h2[:, 2], h2[:, 3],
        x, y,
        x**2, y**2,
        x * y
    ))  # shape: (n, 10)
    
    # Linear output
    prediction = out_features @ params
    
    return prediction
