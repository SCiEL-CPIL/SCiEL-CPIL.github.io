import numpy as np
import time

# EVOLVE-BLOCK-START
def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a linear combination of Gaussian RBFs and polynomials.
    This basis is universal and flexible, allowing better approximation of unknown functions.
    
    Basis functions:
    1. Gaussian RBF centered at -3.0 with width 1.5
    2. Gaussian RBF centered at -1.5 with width 1.5
    3. Gaussian RBF centered at 0.0 with width 1.5
    4. Gaussian RBF centered at 1.5 with width 1.5
    5. Gaussian RBF centered at 3.0 with width 1.5
    6. Constant (1)
    7. x
    8. x^2
    9. x^3
    10. x^4
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit

    x = np.asarray(x, dtype=np.float64)
    
    # 1. Compute Gaussian RBFs
    # Centers: -3, -1.5, 0, 1.5, 3
    # Width (sigma): 1.5
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    sigma = 1.5
    sigma2 = 2.0 * sigma * sigma
    
    # Compute squared distances: (x - c)^2
    # x shape: (N,), centers shape: (5,) -> diff shape: (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    sq_dist = diff * diff
    
    # RBFs: exp(-sq_dist / (2 * sigma^2))
    rbf = np.exp(-sq_dist / sigma2)
    
    # 2. Compute Polynomial Basis
    p0 = np.ones_like(x)
    p1 = x
    p2 = x**2
    p3 = x**3
    p4 = x**4
    
    # 3. Linear Combination
    # params[0:5] correspond to RBFs
    # params[5:10] correspond to polynomials
    pred = (params[0] * rbf[:, 0] + 
            params[1] * rbf[:, 1] + 
            params[2] * rbf[:, 2] + 
            params[3] * rbf[:, 3] + 
            params[4] * rbf[:, 4] + 
            params[5] * p0 + 
            params[6] * p1 + 
            params[7] * p2 + 
            params[8] * p3 + 
            params[9] * p4)
            
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        # Fallback to a simple linear model if numerical issues arise
        return params[5] + params[6] * x

    if time.time() > deadline:
        # If we are close to the deadline, return the current prediction immediately.
        # Since this function is called many times during optimization, 
        # this guard is mostly for safety against unexpected delays.
        pass
        
    return pred
# EVOLVE-BLOCK-END
