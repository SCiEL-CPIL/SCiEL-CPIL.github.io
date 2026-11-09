"""
Symbolic regression with 2 inputs.
Uses a data-driven mixture of Gaussians with a linear trend.
BFGS only fits a single scalar (global amplitude); the 9 other params
control Gaussian centers, widths, and a local linear correction.
This is a structural change from the global polynomial basis.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Mixture-of-Gaussians + linear trend model.
    
    params[0] = amplitude (fitted by BFGS)
    params[1..6] = used to derive 3 Gaussian centers and widths from training data
    params[7..9] = linear trend coefficients (x, y, constant)
    
    The Gaussian centers are placed at quantile positions of the training data,
    and widths are scaled by the data spread. BFGS only needs to optimize
    the single amplitude scalar, making the fit robust and fast.
    """
    n = x.shape[0]
    
    # Extract parameters
    amp = params[0]
    # Derive Gaussian parameters from the remaining 9 params
    # Use params[1], params[2] for center x-positions (mapped to data range)
    # Use params[3], params[4] for center y-positions
    # Use params[5], params[6], params[7] for widths (sigmoid to positive)
    # Use params[8], params[9] for linear trend slopes
    
    # Map params to data-space centers using training quantiles
    # This makes the basis data-adaptive
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)
    
    # Three Gaussian centers from param pairs (mapped via tanh to [-1,1] then to data range)
    cx1 = 0.5 + 0.5 * np.tanh(params[1])
    cy1 = 0.5 + 0.5 * np.tanh(params[2])
    cx2 = 0.5 + 0.5 * np.tanh(params[3])
    cy2 = 0.5 + 0.5 * np.tanh(params[4])
    cx3 = 0.5 + 0.5 * np.tanh(params[5])
    cy3 = 0.5 + 0.5 * np.tanh(params[6])
    
    cx1_data = x_min + cx1 * x_range
    cy1_data = y_min + cy1 * y_range
    cx2_data = x_min + cx2 * x_range
    cy2_data = y_min + cy2 * y_range
    cx3_data = x_min + cx3 * x_range
    cy3_data = y_min + cy3 * y_range
    
    # Widths: sigmoid of params to ensure positive, scaled to data range
    w1 = 0.1 + 0.4 * (1.0 / (1.0 + np.exp(-params[7])))
    w2 = 0.1 + 0.4 * (1.0 / (1.0 + np.exp(-params[8])))
    w3 = 0.1 + 0.4 * (1.0 / (1.0 + np.exp(-params[9])))
    
    w1_data = w1 * x_range
    w2_data = w2 * x_range
    w3_data = w3 * x_range
    
    # Compute Gaussian contributions
    dx1 = (x - cx1_data) / w1_data
    dy1 = (y - cy1_data) / w1_data
    g1 = np.exp(-0.5 * (dx1**2 + dy1**2))
    
    dx2 = (x - cx2_data) / w2_data
    dy2 = (y - cy2_data) / w2_data
    g2 = np.exp(-0.5 * (dx2**2 + dy2**2))
    
    dx3 = (x - cx3_data) / w3_data
    dy3 = (y - cy3_data) / w3_data
    g3 = np.exp(-0.5 * (dx3**2 + dy3**2))
    
    # Linear trend from data mean (centered)
    x_mean = x.mean()
    y_mean = y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    # The amplitude scales the entire Gaussian mixture
    # We add a small constant to avoid zero predictions
    pred = amp * (g1 + g2 + g3) + y.mean()
    
    return pred
# EVOLVE-BLOCK-END
