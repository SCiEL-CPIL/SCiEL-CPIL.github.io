"""
Strategy: Neural Network with 1 hidden layer.
Parameters:
- params[0:5]: Weights from input x to 5 hidden units
- params[5:10]: Weights from 5 hidden units to output y

Architecture:
y = sum_{j=1}^{5} (params[5+j] * tanh(params[j] * x))

This uses exactly 10 parameters.
The tanh activation introduces non-linearity, allowing the model to approximate a wider class of functions than a linear basis.
BFGS will optimize these 10 parameters.
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes y = W_out @ tanh(W_in * x)
    where W_in is a vector of length 5 and W_out is a vector of length 5.
    params[0:5] are the weights for the hidden layer (W_in).
    params[5:10] are the weights for the output layer (W_out).
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Split parameters
    w_in = params[0:5]   # shape (5,)
    w_out = params[5:10] # shape (5,)
    
    # Compute hidden layer activations
    # h_j = tanh(w_in[j] * x)
    # We can do this as a matrix multiplication or element-wise.
    # Let's create a matrix H of shape (5, N) where N is number of points.
    # H[j, i] = tanh(w_in[j] * x[i])
    
    # Efficient way:
    # x is shape (N,)
    # w_in is shape (5,)
    # We want to compute outer product of w_in and x, then tanh.
    # But we need to be careful with shapes.
    # Let's use broadcasting.
    # w_in[:, np.newaxis] * x[np.newaxis, :] -> shape (5, N)
    
    h = np.tanh(w_in[:, np.newaxis] * x[np.newaxis, :])
    
    # Output layer
    # y = w_out @ h
    # w_out is shape (5,)
    # h is shape (5, N)
    # w_out @ h -> shape (N,)
    
    y = np.dot(w_out, h)
    
    return y
