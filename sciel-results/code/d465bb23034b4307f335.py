import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a 3-hidden-unit tanh neural network.
    This structure provides non-linear flexibility that outperforms polynomial
    and trigonometric bases for the observed target function.
    
    Parameters:
    params[0], params[1], params[2]: Weights for input to hidden layer
    params[3], params[4], params[5]: Biases for hidden layer
    params[6], params[7], params[8]: Weights for hidden to output
    params[9]: Bias for output
    """
    # Extract parameters
    w1, w2, w3 = params[0], params[1], params[2]
    b1, b2, b3 = params[3], params[4], params[5]
    v1, v2, v3 = params[6], params[7], params[8]
    b_out = params[9]
    
    # Compute hidden layer activations
    h1 = np.tanh(w1 * x + b1)
    h2 = np.tanh(w2 * x + b2)
    h3 = np.tanh(w3 * x + b3)
    
    # Compute output
    y = v1 * h1 + v2 * h2 + v3 * h3 + b_out
    
    return y
