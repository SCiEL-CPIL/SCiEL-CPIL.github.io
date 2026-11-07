import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a cubic polynomial and two Gaussian RBF bumps.
    
    Model:
    y(x) = c0 + c1*x + c2*x^2 + c3*x^3
           + A1 * exp(-0.5 * ((x - mu1) / sigma1)^2)
           + A2 * exp(-0.5 * ((x - mu2) / sigma2)^2)
    
    Parameters:
    c0-c3: polynomial coefficients
    c4 (A1), c5 (sigma1), c6 (mu1): first Gaussian bump
    c7 (A2), c8 (sigma2), c9 (mu2): second Gaussian bump
    """
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    A1 = params[4]
    sigma1 = np.abs(params[5]) + 1e-4
    mu1 = params[6]
    A2 = params[7]
    sigma2 = np.abs(params[8]) + 1e-4
    mu2 = params[9]
    
    # Polynomial part
    x2 = x * x
    poly = c0 + c1 * x + c2 * x2 + c3 * x2 * x
    
    # Gaussian RBF bumps
    z1 = (x - mu1) / sigma1
    z2 = (x - mu2) / sigma2
    bump1 = A1 * np.exp(-0.5 * z1 * z1)
    bump2 = A2 * np.exp(-0.5 * z2 * z2)
    
    y = poly + bump1 + bump2
    
    return y
