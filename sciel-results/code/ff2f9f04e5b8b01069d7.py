"""Symbolic regression with 3 inputs using a small neural network."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a 2-hidden-unit neural network.
    
    This approach uses a small feedforward neural network with tanh activations.
    Unlike linear basis expansions, this model is non-linear in the parameters,
    allowing it to approximate a wider variety of functional forms, including
    complex interactions and non-monotonic behaviors.
    
    Architecture:
    Input: x, y, z
    Hidden Layer: 2 units with tanh activation
    Output: Linear combination of hidden units
    
    Parameters (10 total):
    p[0:3]  : Weights from x, y, z to hidden unit 1
    p[3:6]  : Weights from x, y, z to hidden unit 2
    p[6]    : Bias for hidden unit 1
    p[7]    : Bias for hidden unit 2
    p[8]    : Weight from hidden unit 1 to output
    p[9]    : Weight from hidden unit 2 to output
    """
    p = params
    
    # Ensure finite values in params
    if not np.all(np.isfinite(p)):
        return np.ones_like(x)
    
    # Extract parameters
    w11, w12, w13 = p[0], p[1], p[2]
    w21, w22, w23 = p[3], p[4], p[5]
    b1 = p[6]
    b2 = p[7]
    v1 = p[8]
    v2 = p[9]
    
    # Compute hidden layer activations
    # Inputs are in [0.5, 2], so no clipping needed for tanh, but ensure finite
    h1_arg = w11 * x + w12 * y + w13 * z + b1
    h2_arg = w21 * x + w22 * y + w23 * z + b2
    
    # Clip arguments to prevent overflow in tanh (though tanh is stable, 
    # very large args just saturate, which is fine, but gradient might vanish.
    # For forward pass, we just need the value.
    # np.tanh is stable for large inputs.
    h1 = np.tanh(h1_arg)
    h2 = np.tanh(h2_arg)
    
    # Compute output
    result = v1 * h1 + v2 * h2
    
    # Ensure output is finite
    if not np.all(np.isfinite(result)):
        return np.ones_like(x)
        
    return result
# EVOLVE-BLOCK-END
