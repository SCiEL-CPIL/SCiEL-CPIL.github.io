import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a single hidden-layer MLP with tanh activation.
    
    The 10 parameters are split into 5 weights for the hidden layer and 5 weights for the
    output layer. The hidden layer has 5 neurons with fixed biases, and the output layer
    has 1 neuron with a fixed bias. This non-linear structure allows the model to capture
    complex functional forms that a linear basis expansion cannot.
    
    The architecture:
    - Hidden layer: h_j = tanh(sum_i w1[i,j] * x + b1[j]) for j=0..4
    - Output layer: y = sum_j w2[j] * h_j + b2
    
    Since the parameters are the weights, BFGS will optimize them directly.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Split 10 params into hidden weights (5x1=5) and output weights (5x1=5)
    # First 5 params: weights from input to hidden layer
    # Last 5 params: weights from hidden layer to output
    
    w1 = params[:5]  # Input to hidden weights
    w2 = params[5:]  # Hidden to output weights
    
    # Fixed biases for the hidden layer
    # Use small biases to avoid symmetry breaking issues
    b1 = np.array([0.1, -0.1, 0.2, -0.2, 0.0])
    
    # Fixed bias for the output layer
    b2 = 0.0
    
    # Compute hidden layer activations
    # x shape: (n_points,)
    # h shape: (n_points, 5)
    z1 = x[:, np.newaxis] * w1[np.newaxis, :] + b1[np.newaxis, :]
    h = np.tanh(z1)
    
    # Compute output
    # y shape: (n_points,)
    y = h @ w2 + b2
    
    return y
