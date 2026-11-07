import numpy as np

# Fixed random seed for deterministic initialization of the fixed network layers
RNG = np.random.default_rng(42)

# Dimensions
INPUT_DIM = 3  # x, y, bias
HIDDEN1_DIM = 12
HIDDEN2_DIM = 9  # 9 weights + 1 bias = 10 params

# Initialize fixed weights for hidden layers
# Scaling to keep activations in sensitive region of tanh
W1 = RNG.normal(0, 1, (HIDDEN1_DIM, INPUT_DIM)) / np.sqrt(INPUT_DIM)
b1 = RNG.normal(0, 1, HIDDEN1_DIM) / np.sqrt(HIDDEN1_DIM)

W2 = RNG.normal(0, 1, (HIDDEN2_DIM, HIDDEN1_DIM)) / np.sqrt(HIDDEN1_DIM)
b2 = RNG.normal(0, 1, HIDDEN2_DIM) / np.sqrt(HIDDEN2_DIM)

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x, y are arrays of shape (N,)
    # params is array of shape (10,)
    
    # Reshape x, y for matrix multiplication
    # Input vector: [x, y, 1]
    # Shape: (N, 3)
    X = np.column_stack((x, y, np.ones_like(x)))
    
    # Hidden Layer 1
    # Z1 = X @ W1.T + b1
    # A1 = tanh(Z1)
    Z1 = X @ W1.T + b1
    A1 = np.tanh(Z1)
    
    # Hidden Layer 2
    # Z2 = A1 @ W2.T + b2
    # A2 = tanh(Z2)
    Z2 = A1 @ W2.T + b2
    A2 = np.tanh(Z2)
    
    # Output Layer
    # The 10 parameters are the weights for A2 and the bias.
    # params[0:9] are weights for the 9 hidden units
    # params[9] is the bias
    weights = params[:HIDDEN2_DIM]
    bias = params[HIDDEN2_DIM]
    
    # Prediction
    pred = A2 @ weights + bias
    
    return pred
