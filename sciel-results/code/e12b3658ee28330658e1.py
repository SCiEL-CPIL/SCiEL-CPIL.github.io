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
    - 5 auxiliary terms: x, v, x^2, v^2, bias
    Total: 5 + 5 = 10 parameters
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
    
    # Widths scaled to domain size / number of centers
    n_centers = 5
    width_t = t_range / n_centers
    width_x = x_range / n_centers
    width_v = v_range / n_centers
    
    # Define 5 RBF centers
    # Spread across time and state space
    t_c1, x_c1, v_c1 = 32.0, -0.10, -0.20
    t_c2, x_c2, v_c2 = 36.0, 0.00, 0.00
    t_c3, x_c3, v_c3 = 40.0, 0.10, 0.10
    t_c4, x_c4, v_c4 = 44.0, -0.10, 0.20
    t_c5, x_c5, v_c5 = 48.0, 0.00, 0.00
    
    # Compute squared distances for each RBF center
    # RBF 1
    dt1 = t - t_c1
    dx1 = x - x_c1
    dv1 = v - v_c1
    dist1_sq = (dt1 / width_t) ** 2 + (dx1 / width_x) ** 2 + (dv1 / width_v) ** 2
    
    # RBF 2
    dt2 = t - t_c2
    dx2 = x - x_c2
    dv2 = v - v_c2
    dist2_sq = (dt2 / width_t) ** 2 + (dx2 / width_x) ** 2 + (dv2 / width_v) ** 2
    
    # RBF 3
    dt3 = t - t_c3
    dx3 = x - x_c3
    dv3 = v - v_c3
    dist3_sq = (dt3 / width_t) ** 2 + (dx3 / width_x) ** 2 + (dv3 / width_v) ** 2
    
    # RBF 4
    dt4 = t - t_c4
    dx4 = x - x_c4
    dv4 = v - v_c4
    dist4_sq = (dt4 / width_t) ** 2 + (dx4 / width_x) ** 2 + (dv4 / width_v) ** 2
    
    # RBF 5
    dt5 = t - t_c5
    dx5 = x - x_c5
    dv5 = v - v_c5
    dist5_sq = (dt5 / width_t) ** 2 + (dx5 / width_x) ** 2 + (dv5 / width_v) ** 2
    
    # Gaussian RBFs
    rbf1 = np.exp(-0.5 * dist1_sq)
    rbf2 = np.exp(-0.5 * dist2_sq)
    rbf3 = np.exp(-0.5 * dist3_sq)
    rbf4 = np.exp(-0.5 * dist4_sq)
    rbf5 = np.exp(-0.5 * dist5_sq)
    
    # Auxiliary terms
    aux_x = x
    aux_v = v
    aux_x2 = x ** 2
    aux_v2 = v ** 2
    aux_bias = 1.0
    
    # Parameter allocation:
    # params[0:5] -> RBF weights
    # params[5]  -> weight for x
    # params[6]  -> weight for v
    # params[7]  -> weight for x^2
    # params[8]  -> weight for v^2
    # params[9]  -> bias
    
    # Final prediction
    pred = (params[0] * rbf1 +
            params[1] * rbf2 +
            params[2] * rbf3 +
            params[3] * rbf4 +
            params[4] * rbf5 +
            params[5] * aux_x +
            params[6] * aux_v +
            params[7] * aux_x2 +
            params[8] * aux_v2 +
            params[9] * aux_bias)
    
    return pred
# EVOLVE-BLOCK-END
