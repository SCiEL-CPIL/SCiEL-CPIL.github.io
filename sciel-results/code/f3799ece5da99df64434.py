import numpy as np
import time

# Module level time tracking for safety, though the evaluator controls the loop
_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x is expected to be in [0.1, 6]
    # Normalize to [-1, 1] for better numerical conditioning
    u = (x - 3.05) / 2.95
    
    # Compute normalized Chebyshev polynomials for better conditioning
    # Each basis function is scaled so that its RMS over the domain is ~1
    # This helps BFGS by balancing gradient magnitudes
    
    y = np.zeros_like(x)
    
    # T_0(u) = 1, RMS = 1
    y += params[0] * 1.0
    
    if len(params) > 1:
        # T_1(u) = u, RMS on [-1,1] = 1/sqrt(3)
        y += params[1] * u * np.sqrt(3.0)
        
        # For n >= 2, T_n has RMS = 1/sqrt(2) on [-1,1] (approximately)
        # But let's compute them and scale appropriately
        T_prev2 = np.ones_like(x)  # T_0
        T_prev1 = u                # T_1
        
        for n in range(2, 10):
            T_curr = 2.0 * u * T_prev1 - T_prev2
            # Scale by sqrt(2) to normalize RMS
            y += params[n] * T_curr * np.sqrt(2.0)
            T_prev2 = T_prev1
            T_prev1 = T_curr
            
    return y
