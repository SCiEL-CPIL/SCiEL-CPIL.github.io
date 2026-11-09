import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Cubic Polynomial + 2 Gaussian Bumps.
    Parameters:
    p0, p1, p2, p9: Coefficients for 1, x, x^2, x^3
    p3, p4, p5: Center, log-width, Amplitude for Gaussian 1
    p6, p7, p8: Center, log-width, Amplitude for Gaussian 2
    """
    # Extract parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Polynomial part: p0 + p1*x + p2*x^2 + p9*x^3
    poly = p0 + p1 * x + p2 * x**2 + p9 * x**3
    
    # Gaussian 1
    # Use exp(p4) for width to ensure positivity and numerical stability
    width1 = np.exp(p4)
    # Avoid division by zero or overflow if width is too small/large
    # Clip width to reasonable range
    width1 = np.clip(width1, 0.1, 10.0)
    gauss1 = p5 * np.exp(-0.5 * ((x - p3) / width1)**2)
    
    # Gaussian 2
    width2 = np.exp(p7)
    width2 = np.clip(width2, 0.1, 10.0)
    gauss2 = p8 * np.exp(-0.5 * ((x - p6) / width2)**2)
    
    return poly + gauss1 + gauss2
