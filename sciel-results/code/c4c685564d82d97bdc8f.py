"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard variables
_START_TIME = time.time()
_DEADLINE = 150.0  # Seconds
_MARGIN = 5.0      # Safety margin in seconds


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a basis of 10 Chebyshev polynomials of the first kind.
    
    Structural Change:
    - Replaces the mixed basis (low-order Chebyshev + Trig + Gaussians) with a pure 
      higher-order Chebyshev basis (degree 0 to 9).
    - Rationale: Chebyshev polynomials provide the best uniform approximation for 
      smooth functions on a bounded interval. Increasing the degree from 4 to 9 
      significantly increases the model's capacity to capture non-linear features 
      while maintaining excellent numerical conditioning for BFGS optimization.
    """
    # Check deadline
    if time.time() > _START_TIME + _DEADLINE - _MARGIN:
        # Return a safe default if we are running out of time
        # The evaluator will likely fail this, but it prevents hanging
        return np.zeros_like(x)

    x = np.asarray(x, dtype=float)
    
    # Map x from [-2, 2] to [-1, 1]
    z = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_9 using recurrence
    # T_0(z) = 1
    # T_1(z) = z
    # T_{n+1}(z) = 2*z*T_n(z) - T_{n-1}(z)
    
    T = np.zeros((10, x.shape[0]), dtype=float)
    
    T[0] = 1.0
    if x.shape[0] > 0:
        T[1] = z
        
    for n in range(2, 10):
        T[n] = 2.0 * z * T[n-1] - T[n-2]
        
    # Combine with parameters
    # params is an array of length 10
    # y = sum(params[i] * T[i])
    y = np.dot(params, T)
    
    return y
# EVOLVE-BLOCK-END
