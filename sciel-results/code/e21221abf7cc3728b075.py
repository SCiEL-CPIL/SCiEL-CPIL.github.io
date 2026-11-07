# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 4 polynomial terms + 3 exponential terms + 3 trig terms.
    All terms are linear in params, making the LSQ problem convex.
    BFGS from any start converges to the global optimum.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    # Allocate basis matrix: 10 columns
    B = np.zeros((n, 10), dtype=np.float64)
    
    # Polynomial terms (degrees 0-3) mapped to [0,1] scale for stability
    t = x / 6.0  # normalize to [~0.017, 1.0]
    B[:, 0] = 1.0
    B[:, 1] = t
    B[:, 2] = t * t
    B[:, 3] = t * t * t
    
    # Exponential terms: exp(a * t) for different rates
    # Use moderate rates to avoid overflow/underflow
    B[:, 4] = np.exp(1.0 * t)
    B[:, 5] = np.exp(2.0 * t)
    B[:, 6] = np.exp(0.5 * t)
    
    # Trig terms: sin and cos with different frequencies
    B[:, 7] = np.sin(1.0 * t)
    B[:, 8] = np.cos(1.0 * t)
    B[:, 9] = np.sin(2.0 * t)
    
    pred = B @ params
    return pred
# EVOLVE-BLOCK-END
