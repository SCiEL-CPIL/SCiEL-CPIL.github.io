"""
Strategy: Adaptive Multi-Scale Fixed RBF Basis.
We use a fixed set of 8 Gaussian basis functions centered at strategic points in [-1, 1]x[-1, 1].
The centers and scales are fixed to ensure the model is linear in the weights.
To improve robustness against local minima and ensure stable BFGS convergence from all-ones,
we parameterize the weights as exp(w_i). This ensures all weights are positive, preventing
destructive interference between basis functions during the initial optimization steps.
The basis includes both broad Gaussians (sigma=0.8) for capturing global trends and
narrow Gaussians (sigma=0.4) for capturing local variations.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target function using a fixed multi-scale RBF basis.
    
    Parameters:
    x, y: Input coordinates.
    params: Array of length 10. 
            params[0:8] are the log-weights for the 8 RBFs.
            params[8:10] are unused (ignored) to maintain interface compatibility, 
            though BFGS may drift them; they have zero gradient.
    
    Returns:
    Prediction array.
    """
    start_time = time.time()
    
    # Fixed centers for the 8 RBFs
    # We want coverage of the [-1, 1]x[-1, 1] domain.
    # 4 broad centers at corners/edges to capture global shape
    # 4 narrow centers at specific points to capture local details
    # 
    # Let's define centers:
    # Broad (sigma=0.8):
    # 0: (0.0, 0.0)
    # 1: (0.7, 0.7)
    # 2: (-0.7, 0.7)
    # 3: (0.7, -0.7)
    # 
    # Narrow (sigma=0.4):
    # 4: (-0.7, -0.7)
    # 5: (0.0, 0.7)
    # 6: (0.0, -0.7)
    # 7: (0.7, 0.0)
    
    cx = np.array([
        0.0,  0.7, -0.7,  0.7,
       -0.7,  0.0,  0.0,  0.7
    ])
    cy = np.array([
        0.0,  0.7,  0.7, -0.7,
       -0.7,  0.7, -0.7,  0.0
    ])
    
    # Scales:
    # First 4 are broad, next 4 are narrow
    sigma = np.array([
        0.8, 0.8, 0.8, 0.8,
        0.4, 0.4, 0.4, 0.4
    ])
    
    # Extract log-weights from params
    # If params are all ones initially, exp(1) ~ 2.718, which is a reasonable positive starting weight.
    # Using exp ensures weights are always positive.
    log_w = params[0:8]
    w = np.exp(log_w)
    
    pred = np.zeros_like(x)
    
    for i in range(8):
        dist_sq = (x - cx[i])**2 + (y - cy[i])**2
        # Calculate Gaussian value
        # exp(-dist_sq / (2 * sigma^2))
        exponent = -dist_sq / (2.0 * sigma[i]**2)
        # Clip exponent to prevent underflow issues in extreme cases (though exp(-large) is just 0)
        # and overflow (which shouldn't happen as exponent is <= 0)
        pred += w[i] * np.exp(exponent)
        
    # Safety check for time
    if time.time() - start_time > 110:
        # Fallback to simple linear model if computation is unexpectedly slow
        # This is just a safety net; the RBF loop is O(N*8) which is very fast
        return params[0] * x + params[1] * y + params[2]
        
    return pred
# EVOLVE-BLOCK-END
