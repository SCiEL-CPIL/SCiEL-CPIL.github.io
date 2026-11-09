"""
Strategy: Small Deep Neural Network (MLP) with Non-linear Activations.

This strategy departs from the linear-in-parameters basis functions (Chebyshev/Fourier/RBF)
by using a small Multi-Layer Perceptron (MLP). 
Rationale:
1. Universal Approximation: Unlike fixed smooth bases, an MLP with non-linear activations 
   (e.g., tanh or ReLU) can approximate any continuous function, including non-smooth ones 
   (like |x| or step-like transitions) that Chebyshev polynomials struggle with.
2. Parameter Allocation: We have 10 parameters. We structure them as:
   - 3 weights for the input-to-hidden layer (3 inputs? No, 1 input x). 
   Wait, 1 input x. 
   Let's design a very small net to fit exactly 10 params.
   
   Architecture:
   Input: x (scalar per point)
   Hidden Layer 1: 2 neurons
   Output Layer: 1 neuron
   
   Parameters:
   - Weights W1 (2x1): w11, w12
   - Biases b1 (2x1): b11, b12
   - Weights W2 (1x2): w21, w22
   - Biases b2 (1x1): b21
   Total so far: 2 + 2 + 2 + 1 = 7 params.
   We have 3 left. Let's add another hidden layer or more neurons.
   
   Let's try:
   Input: x
   Hidden 1: 3 neurons
   Hidden 2: 2 neurons
   Output: 1 neuron
   
   Params:
   W1 (3x1): 3
   b1 (3x1): 3
   W2 (2x3): 6
   b2 (2x1): 2
   W3 (1x2): 2
   b3 (1x1): 1
   Total: 3+3+6+2+2+1 = 17. Too many.
   
   Let's stick to 10 params exactly.
   Architecture:
   Input: x
   Hidden 1: 2 neurons (tanh activation)
   Output: 1 neuron (linear)
   
   Params:
   w11, w12 (weights from x to h1, h2) -> 2
   b11, b12 (biases for h1, h2) -> 2
   w21, w22 (weights from h1, h2 to out) -> 2
   b21 (bias for out) -> 1
   Total: 7.
   
   Remaining 3 params? We can use them to scale the input or add a second hidden layer with 1 neuron?
   Or just use a slightly larger net and share some structure? No, params are a flat array.
   
   Let's use 4 hidden neurons in 1 layer?
   W1 (4x1): 4
   b1 (4x1): 4
   W2 (1x4): 4
   b2 (1x1): 1
   Total: 13. Too many.
   
   How about 3 hidden neurons?
   W1 (3x1): 3
   b1 (3x1): 3
   W2 (1x3): 3
   b2 (1x1): 1
   Total: 10. Perfect.
   
   So:
   h1 = tanh(w1*x + b1)
   h2 = tanh(w2*x + b2)
   h3 = tanh(w3*x + b3)
   y = v1*h1 + v2*h2 + v3*h3 + c
   
   Params:
   p[0]=w1, p[1]=w2, p[2]=w3
   p[3]=b1, p[4]=b2, p[5]=b3
   p[6]=v1, p[7]=v2, p[8]=v3
   p[9]=c
   
   This is a 3-neuron tanh MLP. It is non-linear and can fit complex shapes.
   BFGS might struggle with non-convexity, but starting from all ones is a specific initialization.
   To help BFGS, we can add a small regularization or just rely on the smoothness of tanh.
   However, tanh can saturate. ReLU is not differentiable at 0, which might break BFGS (which uses gradients).
   So tanh is safer for BFGS.
   
   To improve conditioning, we might scale x. x is in [-3, 3].
   Let's use x/3 as input to the network to keep values in [-1, 1].
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the output of a 3-neuron Tanh MLP.
    
    Args:
        x: Input array of shape (N,)
        params: Coefficients array of shape (10,)
            [w1, w2, w3, b1, b2, b3, v1, v2, v3, c]
    
    Returns:
        Prediction array of shape (N,)
    """
    x = np.asarray(x).ravel()
    
    # Extract parameters
    w1, w2, w3 = params[0], params[1], params[2]
    b1, b2, b3 = params[3], params[4], params[5]
    v1, v2, v3 = params[6], params[7], params[8]
    c = params[9]
    
    # Scale input to [-1, 1] for better conditioning
    t = x / 3.0
    
    # Hidden layer: 3 tanh neurons
    h1 = np.tanh(w1 * t + b1)
    h2 = np.tanh(w2 * t + b2)
    h3 = np.tanh(w3 * t + b3)
    
    # Output layer: linear combination
    y = v1 * h1 + v2 * h2 + v3 * h3 + c
    
    return y

# EVOLVE-BLOCK-END
