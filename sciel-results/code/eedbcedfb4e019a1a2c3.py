"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using a small Neural Network (MLP).
    
    Strategy:
    Instead of a fixed linear basis (like RBFs or Polynomials), we interpret the 10 
    parameters as the weights of a small neural network. Specifically, we use a 
    single-layer network with 10 hidden units (tanh activation) and a single output 
    unit. The parameters p[0:10] are the input-to-hidden weights. The hidden-to-output 
    weights are fixed to 1.0 to keep the problem well-posed and avoid overfitting 
    with only 10 parameters.
    
    Architecture:
    1. Input x is normalized to [-1, 1] by dividing by 3 (since x in [-3, 3]).
    2. Hidden layer: z = tanh(x_norm * p[0:10])
    3. Output: y = sum(z)
    
    This creates a function of the form:
    y(x) = sum_{i=0}^{9} tanh( (x/3) * p_i )
    
    This representation is universal for continuous functions and can approximate 
    step-like features, oscillations, and smooth curves. The nonlinearity of tanh 
    allows it to capture interactions that a linear combination of fixed basis 
    functions cannot. BFGS should handle the smooth gradient landscape of this 
    small network well.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Ensure p has length 10
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    else:
        p = p[:10]
        
    # Normalize input to roughly [-1, 1] to stabilize tanh behavior
    x_norm = x / 3.0
    
    # Compute hidden layer activations
    # z shape: (N, 10)
    # x_norm[:, np.newaxis] shape: (N, 1)
    # p[np.newaxis, :] shape: (1, 10)
    z = np.tanh(x_norm[:, np.newaxis] * p[np.newaxis, :])
    
    # Output is sum of hidden units (weights fixed to 1)
    pred = np.sum(z, axis=1)
    
    # Safety check
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
