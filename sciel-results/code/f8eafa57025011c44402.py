"""
Strategy: Single Hidden Layer Neural Network (MLP).
Instead of a global polynomial or fixed Gaussian bumps, we use a learnable
weighted sum of non-linear activations. This allows the model to capture
non-linear features (e.g., kinks, saturation) that a smooth Chebyshev series
might approximate poorly, while remaining differentiable for BFGS.

Params layout (length 10):
params[0:5] -> Weights for the 5 hidden units (input to hidden)
params[5:10] -> Biases for the 5 hidden units
The output is a fixed linear combination of the hidden activations with
weights initialized to 1.0 (implicit in the sum) to keep the parameter count
at exactly 10. To allow the output scale to adjust, we include a final
scaling factor derived from the mean of the hidden weights, or simply
rely on the BFGS optimizer to find the optimal balance. 
Actually, to strictly use 10 params and allow output scaling, we can interpret
the 10 params as:
5 weights (w_i) and 5 biases (b_i).
Output y = sum_i tanh(w_i * x + b_i) / 5.
The division by 5 keeps the magnitude reasonable.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a single hidden layer neural network.
    
    params[0:5] are the input weights w_i.
    params[5:10] are the biases b_i.
    
    y = (1/5) * sum_{i=1}^{5} tanh(w_i * x + b_i)
    """
    # Extract parameters
    w = params[0:5]
    b = params[5:10]
    
    # Compute hidden layer activations
    # x shape: (N, 1) or (N,)
    # We need to broadcast w and b correctly.
    # If x is (N,), w is (5,), b is (5,)
    # We want (N, 5) matrix where each column i is w[i]*x + b[i]
    
    # Ensure x is a column vector for easy broadcasting if needed, 
    # but standard numpy broadcasting works if we reshape.
    x_reshaped = x.reshape(-1, 1) # (N, 1)
    w_reshaped = w.reshape(1, 5)  # (1, 5)
    b_reshaped = b.reshape(1, 5)  # (1, 5)
    
    # z = x * w + b -> (N, 5)
    z = x_reshaped * w_reshaped + b_reshaped
    
    # Activation
    a = np.tanh(z)
    
    # Output: average of the hidden units
    # This scaling helps prevent gradient vanishing/exploding during BFGS
    result = np.mean(a, axis=1)
    
    return result
# EVOLVE-BLOCK-END
