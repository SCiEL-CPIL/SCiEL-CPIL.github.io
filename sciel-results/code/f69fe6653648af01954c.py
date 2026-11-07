import numpy as np

# Fixed weights for the neural network basis
# We use a fixed seed to ensure the basis is deterministic and reproducible
# across evaluations, which is crucial for the optimizer to converge consistently.
# The "params" are the only adjustable parameters, acting as the final layer weights.

# Define the architecture
# Input: 2 features (x, y)
# Hidden Layer 1: 32 tanh units
# Hidden Layer 2: 10 tanh units (matches the 10 params)
# Output: Linear combination of Hidden Layer 2 outputs

# Initialize weights with a fixed seed
np.random.seed(42)

# Weights for layer 1 (Input -> Hidden 1)
# Shape: (32, 2)
# Scale: For input in [-1, 1], std 1.0 is reasonable.
W1 = np.random.randn(32, 2)
b1 = np.zeros(32)

# Weights for layer 2 (Hidden 1 -> Hidden 2)
# Shape: (10, 32)
# Scale: Input to this layer is tanh output, in [-1, 1].
# To keep pre-activations in a non-saturated range, scale by 1/sqrt(32).
W2 = np.random.randn(10, 32) / np.sqrt(32)
b2 = np.zeros(10)

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a fixed-weight tanh neural network basis.
    
    The 10 parameters are the weights of the final linear output layer.
    The internal structure (W1, b1, W2, b2) is fixed.
    
    Args:
        x: Input feature 1, array of shape (N,)
        y: Input feature 2, array of shape (N,)
        params: Array of 10 parameters, shape (10,)
        
    Returns:
        Predictions, array of shape (N,)
    """
    # Stack inputs into a matrix of shape (N, 2)
    X = np.column_stack((x, y))
    
    # Hidden Layer 1
    Z1 = X @ W1.T + b1
    A1 = np.tanh(Z1)
    
    # Hidden Layer 2
    Z2 = A1 @ W2.T + b2
    A2 = np.tanh(Z2)
    
    # Output Layer
    # A2 has shape (N, 10), params has shape (10,)
    # Result is (N,)
    f = A2 @ params
    
    return f
