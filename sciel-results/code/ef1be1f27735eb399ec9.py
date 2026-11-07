import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a physically-motivated polynomial basis.
    The basis is designed to capture Van der Pol dynamics (limit cycle oscillator)
    with amplitude-dependent damping and nonlinear restoring forces.
    
    Basis terms (using scaled inputs X, V):
    1. 1 (bias)
    2. X (linear restoring)
    3. V (linear damping)
    4. X^2 (quadratic nonlinearity)
    5. X*V (cross term)
    6. V^2 (quadratic velocity)
    7. X^3 (cubic restoring)
    8. X^2*V (Van der Pol damping term)
    9. V^3 (cubic damping)
    10. X^4 (quartic restoring)
    
    Inputs are scaled to roughly [-1, 1] to improve conditioning for BFGS.
    """
    # Time guard
    start_time = time.time()
    timeout = 140.0 # Safety margin before 150s limit
    
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to improve numerical stability
    # Ranges from task: x [-0.5728, 0.5275], v [-0.1754, 0.2821]
    # Midpoints: x ~ -0.02, v ~ 0.05
    # Widths: x ~ 1.1, v ~ 0.46
    # We scale by width/2 to get approx [-1, 1]
    x_mean = -0.02
    x_scale = 0.55
    v_mean = 0.05
    v_scale = 0.23
    
    X = (x - x_mean) / x_scale
    V = (v - v_mean) / v_scale
    
    # Compute basis terms
    # We compute these explicitly to allow BFGS to optimize the linear combination
    # This is linear in params, so the cost function is quadratic.
    
    b0 = np.ones_like(X)
    b1 = X
    b2 = V
    b3 = X * X
    b4 = X * V
    b5 = V * V
    b6 = X * X * X
    b7 = X * X * V
    b8 = V * V * V
    b9 = X * X * X * X
    
    # Combine
    # Using dot product or explicit sum
    # Explicit sum is faster for small fixed number of terms
    result = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 + 
              p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
    
    # Check time
    if time.time() - start_time > timeout:
        # Return current best (which is just the result) 
        # Since we don't have iterative search inside, this is just a safety check
        # If this function is called many times, the overhead is low.
        pass
        
    return result
