# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate: A small 2-layer neural network (MLP) with 2 hidden neurons
    and a linear skip connection.
    
    Architecture:
    Hidden Layer:
      h1 = tanh(w1*x + w2*y + w3)
      h2 = tanh(w4*x + w5*y + w6)
    Output Layer:
      pred = w7*h1 + w8*h2 + w9*x + w10*y
    
    Parameters (10 total):
      w1, w2, w3: Weights and bias for hidden neuron 1
      w4, w5, w6: Weights and bias for hidden neuron 2
      w7, w8: Weights for hidden neurons in output
      w9, w10: Weights for direct linear input connections
    
    This model is a universal approximator (given enough neurons) and can capture both
    global linear trends and local non-linear structures. The tanh activation provides
    smooth non-linearity. BFGS should be able to fit this well for many functions.
    """
    w1, w2, w3, w4, w5, w6, w7, w8, w9, w10 = params
    
    # Compute hidden layer activations
    z1 = w1 * x + w2 * y + w3
    h1 = np.tanh(z1)
    
    z2 = w4 * x + w5 * y + w6
    h2 = np.tanh(z2)
    
    # Compute output as linear combination of hidden units and direct inputs
    pred = w7 * h1 + w8 * h2 + w9 * x + w10 * y
    
    return pred
# EVOLVE-BLOCK-END
