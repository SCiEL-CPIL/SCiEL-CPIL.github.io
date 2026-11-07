"""Gaussian Process-inspired kernel basis for nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled as a sum of Gaussian basis functions with independent scales.
    
    Parameters:
        t: time (input 1)
        x: position (input 2)
        v: velocity (input 3)
        params: array of 10 parameters.
                params[0:3] = center 1 (t_c1, x_c1, v_c1)
                params[3]   = weight 1 (w1)
                params[4]   = scale 1 (inverse bandwidth, transformed via softplus)
                params[5:8] = center 2 (t_c2, x_c2, v_c2)
                params[8]   = weight 2 (w2)
                params[9]   = scale 2 (inverse bandwidth, transformed via softplus)
    
    Returns:
        Acceleration predictions (same shape as t)
    """
    # Extract parameters
    t_c1, x_c1, v_c1 = params[0], params[1], params[2]
    w1 = params[3]
    scale1 = np.log1p(np.exp(params[4]))  # softplus for positive scale
    
    t_c2, x_c2, v_c2 = params[5], params[6], params[7]
    w2 = params[8]
    scale2 = np.log1p(np.exp(params[9]))  # softplus for positive scale
    
    # Compute Gaussian basis functions
    # g(t, x, v) = exp(-scale * ((t - t_c)^2 + (x - x_c)^2 + (v - v_c)^2))
    
    dt1 = t - t_c1
    dx1 = x - x_c1
    dv1 = v - v_c1
    r2_1 = dt1**2 + dx1**2 + dv1**2
    g1 = np.exp(-scale1 * r2_1)
    
    dt2 = t - t_c2
    dx2 = x - x_c2
    dv2 = v - v_c2
    r2_2 = dt2**2 + dx2**2 + dv2**2
    g2 = np.exp(-scale2 * r2_2)
    
    # Weighted sum
    # Note: No explicit bias. The wide Gaussian (low scale) can approximate the mean/bias.
    # This utilizes all 10 parameters for more flexible shape control.
    
    return w1 * g1 + w2 * g2
# EVOLVE-BLOCK-END
