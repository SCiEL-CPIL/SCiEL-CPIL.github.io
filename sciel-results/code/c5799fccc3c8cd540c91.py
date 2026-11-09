"""RBF network for nonlinear oscillator acceleration."""
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using 2 Gaussian RBFs + linear baseline.
    
    Parameters:
        x: position array
        v: velocity array
        params: array of 10 values
            p[0] = RBF1 center_x
            p[1] = RBF1 center_v
            p[2] = RBF1 width
            p[3] = RBF1 weight
            p[4] = RBF2 center_x
            p[5] = RBF2 center_v
            p[6] = RBF2 width
            p[7] = RBF2 weight
            p[8] = linear x coefficient
            p[9] = linear v coefficient
    """
    # Ensure widths are positive and not too small to avoid numerical issues
    w1 = np.abs(params[2]) + 1e-4
    w2 = np.abs(params[6]) + 1e-4
    
    # RBF1: Gaussian centered at (p0, p1) with width p2, weight p3
    dx1 = x - params[0]
    dv1 = v - params[1]
    rbf1 = params[3] * np.exp(-(dx1 * dx1 + dv1 * dv1) / (w1 * w1))
    
    # RBF2: Gaussian centered at (p4, p5) with width p6, weight p7
    dx2 = x - params[4]
    dv2 = v - params[5]
    rbf2 = params[7] * np.exp(-(dx2 * dx2 + dv2 * dv2) / (w2 * w2))
    
    # Linear baseline
    linear = params[8] * x + params[9] * v
    
    return linear + rbf1 + rbf2
