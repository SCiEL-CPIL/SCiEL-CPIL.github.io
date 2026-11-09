# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using Gaussian RBF basis on (x, v)
    with local linear modulation to capture phase-plane coupling.
    
    Structural change from global polynomials: uses a small set of fixed
    Gaussian RBF centers on a 2D grid of (x, v), where each basis function
    multiplies a linear term (1 + x + v) to approximate the acceleration
    field locally. This captures non-additive nonlinear coupling more
    effectively than global polynomials while maintaining 10 parameters.
    
    Inputs:
    t: Time
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
    
    Basis Functions:
    5 RBF centers on a 2D grid of (x, v) with width sigma.
    Each RBF is modulated by (1 + alpha*x + beta*v) to allow local linear
    variation around each center.
    
    p0-p4: Weights for each RBF center (5 parameters)
    p5: RBF width (sigma)
    p6: Linear modulation coefficient for x
    p7: Linear modulation coefficient for v
    p8: Additional global linear x term
    p9: Additional global linear v term
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # RBF centers on a 2D grid in (x, v) space
    # x range: [-0.2056, 0.1749], center ~ -0.015, half-range ~ 0.19
    # v range: [-0.3391, 0.3624], center ~ 0.012, half-range ~ 0.35
    
    # 5 centers: 4 corners + 1 center of the (x, v) domain
    x_centers = np.array([-0.15, 0.10, -0.15, 0.10, -0.015])
    v_centers = np.array([-0.25, -0.25, 0.25, 0.25, 0.012])
    
    # Ensure sigma is positive and reasonable
    sigma = np.abs(p5) + 0.01
    sigma = np.clip(sigma, 0.05, 1.0)
    
    # Compute RBF values for each center
    # For each center i, compute exp(-((x - xi)^2 + (v - vi)^2) / (2*sigma^2))
    rbf_vals = np.zeros((5,) + x.shape, dtype=np.float64)
    for i in range(5):
        dx = x - x_centers[i]
        dv = v - v_centers[i]
        rbf_vals[i] = np.exp(-(dx * dx + dv * dv) / (2.0 * sigma * sigma))
    
    # Local linear modulation: (1 + p6 * x + p7 * v)
    modulation = 1.0 + p6 * x + p7 * v
    
    # Weighted sum of modulated RBFs
    result = (p0 * rbf_vals[0] * modulation +
              p1 * rbf_vals[1] * modulation +
              p2 * rbf_vals[2] * modulation +
              p3 * rbf_vals[3] * modulation +
              p4 * rbf_vals[4] * modulation)
    
    # Add global linear terms for overall trend
    result = result + p8 * x + p9 * v
    
    return result
# EVOLVE-BLOCK-END
