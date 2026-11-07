"""RBF-based symbolic regression for nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using a Gaussian RBF basis.
    
    Inputs:
        t: time, range [30, 50]
        x: position, range [-0.2056, 0.1749]
        v: velocity, range [-0.3391, 0.3624]
        params: array of 10 fitted constants
    
    The RBF network uses:
    - 5 centers in the 3D input space (t, x, v)
    - 5 RBF weights
    - 2 linear terms (x, v)
    - 2 quadratic terms (x^2, v^2)
    - 1 bias term
    Total: 5 + 2 + 2 + 1 = 10 parameters
    """
    # Ensure inputs are arrays
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    # Input ranges
    t_min, t_max = 30.0, 50.0
    x_min, x_max = -0.2056, 0.1749
    v_min, v_max = -0.3391, 0.3624
    
    # Domain sizes
    t_range = t_max - t_min  # 20
    x_range = x_max - x_min  # ~0.38
    v_range = v_max - v_min  # ~0.70
    
    # Widths scaled to domain size
    # Using 2.5 as divisor gives moderate overlap
    width_t = t_range / 2.5
    width_x = x_range / 2.5
    width_v = v_range / 2.5
    
    # Define 5 RBF centers
    # Spread across the domain
    # Center 1: Early time, neg x, neg v
    t_c1, x_c1, v_c1 = 35.0, -0.10, -0.20
    # Center 2: Mid time, origin
    t_c2, x_c2, v_c2 = 40.0, 0.00, 0.00
    # Center 3: Late time, pos x, pos v
    t_c3, x_c3, v_c3 = 45.0, 0.10, 0.20
    # Center 4: Early time, pos x, pos v
    t_c4, x_c4, v_c4 = 35.0, 0.10, 0.20
    # Center 5: Late time, neg x, neg v
    t_c5, x_c5, v_c5 = 45.0, -0.10, -0.20
    
    # Compute squared distances for each RBF center
    # RBF 1
    dt1 = (t - t_c1) / width_t
    dx1 = (x - x_c1) / width_x
    dv1 = (v - v_c1) / width_v
    dist1_sq = dt1**2 + dx1**2 + dv1**2
    
    # RBF 2
    dt2 = (t - t_c2) / width_t
    dx2 = (x - x_c2) / width_x
    dv2 = (v - v_c2) / width_v
    dist2_sq = dt2**2 + dx2**2 + dv2**2
    
    # RBF 3
    dt3 = (t - t_c3) / width_t
    dx3 = (x - x_c3) / width_x
    dv3 = (v - v_c3) / width_v
    dist3_sq = dt3**2 + dx3**2 + dv3**2
    
    # RBF 4
    dt4 = (t - t_c4) / width_t
    dx4 = (x - x_c4) / width_x
    dv4 = (v - v_c4) / width_v
    dist4_sq = dt4**2 + dx4**2 + dv4**2
    
    # RBF 5
    dt5 = (t - t_c5) / width_t
    dx5 = (x - x_c5) / width_x
    dv5 = (v - v_c5) / width_v
    dist5_sq = dt5**2 + dx5**2 + dv5**2
    
    # Gaussian RBFs
    rbf1 = np.exp(-0.5 * dist1_sq)
    rbf2 = np.exp(-0.5 * dist2_sq)
    rbf3 = np.exp(-0.5 * dist3_sq)
    rbf4 = np.exp(-0.5 * dist4_sq)
    rbf5 = np.exp(-0.5 * dist5_sq)
    
    # Polynomial terms for local structure
    lin_x = x
    lin_v = v
    quad_x2 = x ** 2
    quad_v2 = v ** 2
    
    # Parameter allocation:
    # params[0:5] -> RBF weights
    # params[5]   -> linear weight for x
    # params[6]   -> linear weight for v
    # params[7]   -> quadratic weight for x^2
    # params[8]   -> quadratic weight for v^2
    # params[9]   -> bias
    
    # Final prediction
    pred = (params[0] * rbf1 +
            params[1] * rbf2 +
            params[2] * rbf3 +
            params[3] * rbf4 +
            params[4] * rbf5 +
            params[5] * lin_x +
            params[6] * lin_v +
            params[7] * quad_x2 +
            params[8] * quad_v2 +
            params[9] * 1.0)
    
    return pred
# EVOLVE-BLOCK-END
