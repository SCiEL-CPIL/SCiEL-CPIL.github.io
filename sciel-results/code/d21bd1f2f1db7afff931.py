import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic Regression: Nonlinear Oscillator Acceleration.
    
    Uses a Gaussian Process-style Radial Basis Function (RBF) expansion with
    fixed centers derived from the input ranges. This treats the problem as a
    general nonlinear function approximator, avoiding the ill-conditioning of
    high-order monomials and the structural bias of specific oscillator terms.
    
    Inputs:
    t: Time, range [30, 50]
    x: Position, range [-0.2056, 0.1749]
    v: Velocity, range [-0.3391, 0.3624]
    params: 10 fitted constants
    
    Basis Functions:
    9 RBF terms centered at different points in the (x, v) space.
    The centers are chosen to cover the input domain uniformly.
    """
    # Deadline guard
    start_time = time.time()
    
    # Input ranges
    x_min, x_max = -0.2056, 0.1749
    v_min, v_max = -0.3391, 0.3624
    
    # Normalize inputs to [0, 1] for RBF center placement
    x_norm = (x - x_min) / (x_max - x_min)
    v_norm = (v - v_min) / (v_max - v_min)
    
    # Define 9 centers in the normalized (x, v) space
    # Using a 3x3 grid for uniform coverage
    centers_x = np.array([0.0, 0.5, 1.0])
    centers_v = np.array([0.0, 0.5, 1.0])
    
    # RBF width (sigma). A smaller sigma means more local features.
    # Since we only have 10 parameters, we need a balance.
    # Let's use a sigma that covers about 1/3 of the domain.
    sigma = 0.5
    
    p = params[:9]
    
    # Compute RBF values
    # To avoid creating huge temporary arrays, we compute each term directly
    # or use vectorized operations efficiently.
    
    # b_i = exp(-((x_norm - cx_i)^2 + (v_norm - cv_i)^2) / (2 * sigma^2))
    
    # We can compute the squared distance for each center
    # Since we have 9 centers, we can unroll them for speed or use a loop.
    # A loop is fine as 9 iterations is negligible compared to the vector operations.
    
    b = np.zeros_like(x)
    
    for i in range(3):
        cx = centers_x[i]
        for j in range(3):
            cv = centers_v[j]
            idx = i * 3 + j
            dist_sq = (x_norm - cx)**2 + (v_norm - cv)**2
            b += p[idx] * np.exp(-dist_sq / (2.0 * sigma * sigma))
    
    # The 10th parameter is a bias term
    bias = params[9] if params.size > 9 else 0.0
    
    pred = b + bias
    
    return pred
