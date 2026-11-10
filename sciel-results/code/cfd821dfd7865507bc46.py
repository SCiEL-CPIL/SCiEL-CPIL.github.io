"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Piecewise cubic B-spline with 10 control points.
    
    Uses uniform knots over [-2, 2] with 10 internal knots, giving
    10 control points that directly map to curve values at knot locations.
    This local basis avoids global polynomial bias and ill-conditioning.
    
    Parameters:
    x: Input array of shape (N,)
    params: Array of 10 control point values
    """
    # 10 control points at uniform knots from -2 to 2
    # Knots: -2, -1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6, 2.0 (11 knots, 10 intervals)
    # But we need 10 control points. Let's use 9 internal knots + 2 endpoints = 11 knots
    # Actually, for cubic B-spline with n control points, we need n+3 knots.
    # With 10 control points, we need 13 knots.
    
    # Simpler approach: use cubic interpolation on 10 evenly spaced points
    # Points at: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, and one more?
    # Let's use 10 points from -2 to 2: spacing = 4/9
    n_ctrl = 10
    ctrl_x = np.linspace(-2.0, 2.0, n_ctrl)
    
    # For each x, find which interval it falls in and do cubic Hermite interpolation
    # or use natural cubic spline basis functions
    
    # Better: use the de Boor algorithm or direct B-spline evaluation
    # For simplicity and speed, use numpy interp with cubic mode if available,
    # or implement a simple cubic basis
    
    # Actually, let's use a different structural approach:
    # Sum of localized basis functions (like RBF or local polynomials)
    
    # Use 10 Gaussian-like bumps centered at evenly spaced points
    # y(x) = sum_i params[i] * exp(-((x - c_i)^2) / (2*sigma^2))
    # This is a local basis that can fit arbitrary smooth shapes
    
    centers = np.linspace(-2.0, 2.0, 10)
    sigma = 0.5  # Fixed width for the bumps
    
    # Compute: y(x) = sum_i params[i] * exp(-((x - c_i)^2) / (2*sigma^2))
    # Vectorized: x has shape (N,), centers has shape (10,)
    # diff has shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    # Ensure x is 1D
    if x.ndim == 0:
        x = x.reshape(1)
        diff = (x - centers)[np.newaxis, :]
    
    # Gaussian basis: exp(-0.5 * (diff/sigma)^2)
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    # basis shape: (N, 10)
    
    # y = basis @ params
    result = basis @ params
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
