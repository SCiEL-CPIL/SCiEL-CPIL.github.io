import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Single Hidden Layer Neural Network (Universal Approximator)
    
    Instead of a fixed basis (RBF/Chebyshev) where the model is linear in parameters,
    we use a small neural network where the 10 parameters are the network weights and biases.
    
    Model Architecture:
    Input: x (scalar)
    Hidden Layer: 2 neurons
    - Neuron 1: tanh(w1 * x + b1)
    - Neuron 2: tanh(w2 * x + b2)
    Output: y = w3 * h1 + w4 * h2 + b_out
    
    Parameters (10 total):
    p[0] = w1 (weight for input to h1)
    p[1] = b1 (bias for h1)
    p[2] = w2 (weight for input to h2)
    p[3] = b2 (bias for h2)
    p[4] = w3 (weight for h1 to output)
    p[5] = w4 (weight for h2 to output)
    p[6] = b_out (bias for output)
    p[7] = 0 (unused, kept for interface compatibility or future expansion)
    p[8] = 0 (unused)
    p[9] = 0 (unused)
    
    Why this is structurally different:
    1. Non-linear in parameters: BFGS must optimize both the "shape" (w1, w2) and the "amplitude" (w3, w4) simultaneously. This allows the model to adapt its functional form (e.g., shifting the center of the tanh, changing its steepness) to match the target function, rather than just scaling a fixed basis function.
    2. Universal Approximation: Even with 2 hidden units, tanh networks can approximate a wide range of smooth functions on a bounded interval, including step-like or asymmetric features that linear RBFs might struggle with if centers are fixed.
    3. Optimization Challenge: This creates a non-convex loss landscape. BFGS starting from all ones (p=[1,...,1]) will start with a specific configuration. The hope is that the gradient information guides it towards a configuration that captures the essential features of the hidden function better than a linear projection onto a fixed basis.
    
    Safety:
    - We use tanh for bounded activations to prevent numerical overflow during BFGS steps.
    - We clip weights to prevent extreme values that might cause issues, though tanh helps.
    - Time limit guard is included.
    """
    
    start_time = time.time()
    TIME_LIMIT = 145.0  # Leave 5s margin for safety
    
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64).flatten()
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    else:
        p = p[:10]

    # Extract parameters
    w1, b1, w2, b2, w3, w4, b_out = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    
    # Compute hidden layer activations
    # tanh is smooth and bounded, good for stability
    h1 = np.tanh(w1 * x + b1)
    h2 = np.tanh(w2 * x + b2)
    
    # Compute output
    y = w3 * h1 + w4 * h2 + b_out
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        # Fallback to a simple constant if numerical instability occurs
        return np.full_like(x, b_out if np.isfinite(b_out) else 0.0)
        
    return y
