"""Symbolic regression with 1 input - Small Tanh Neural Network."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Small Tanh Neural Network with 1 hidden layer and 3 nodes.
    
    Model: y = w0 + w1*tanh(v1*x + b1) + w2*tanh(v2*x + b2) + w3*tanh(v3*x + b3)
    
    Parameters:
      params[0]: output bias w0
      params[1]: weight w1 for node 1
      params[2]: weight w2 for node 2
      params[3]: weight w3 for node 3
      params[4]: input slope v1 for node 1
      params[5]: input slope v2 for node 2
      params[6]: input slope v3 for node 3
      params[7]: hidden bias b1 for node 1
      params[8]: hidden bias b2 for node 2
      params[9]: hidden bias b3 for node 3
    """
    w0 = params[0]
    w1, w2, w3 = params[1], params[2], params[3]
    v1, v2, v3 = params[4], params[5], params[6]
    b1, b2, b3 = params[7], params[8], params[9]
    
    # Compute tanh activations
    # Using np.tanh for stability and smoothness
    h1 = np.tanh(v1 * x + b1)
    h2 = np.tanh(v2 * x + b2)
    h3 = np.tanh(v3 * x + b3)
    
    # Linear combination
    y = w0 + w1 * h1 + w2 * h2 + w3 * h3
    
    return y
# EVOLVE-BLOCK-END
