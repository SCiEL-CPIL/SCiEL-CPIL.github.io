"""Rewritten program using an RBF representation."""
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration modeled by a sum of 10 Gaussian Radial Basis Functions.
    The 10 parameters are the weights of the RBFs.
    Centers and widths are fixed to provide a smooth, localized basis.
    """
    # Fixed centers in the (x, v) plane
    # x range approx [-0.6, 0.6], v range approx [-0.2, 0.3]
    xc = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    vc = np.array([-0.1, 0.1])
    
    # Generate 10 centers: 5 in x, 2 in v
    Xc, Vc = np.meshgrid(xc, vc, indexing='ij')
    Xc = Xc.flatten()
    Vc = Vc.flatten()
    
    # Weights from params
    w = params
    
    # Fixed anisotropic widths
    sigma_x = 0.25
    sigma_v = 0.15
    
    # Compute RBF values
    # x, v are (N, 1) or (N,)
    # Ensure 2D for broadcasting
    x = np.atleast_2d(x).T
    v = np.atleast_2d(v).T
    
    dx = x - Xc
    dv = v - Vc
    
    # Exponent: -0.5 * (dx^2/sigma_x^2 + dv^2/sigma_v^2)
    exp_term = -0.5 * (dx**2 / (sigma_x**2) + dv**2 / (sigma_v**2))
    
    rbf_values = np.exp(exp_term)
    
    # Linear combination
    result = rbf_values @ w
    
    # Return 1D array
    return result.flatten()
