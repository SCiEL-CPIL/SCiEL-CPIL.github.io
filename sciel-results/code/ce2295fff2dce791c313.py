import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a Gaussian RBF basis with fixed centers.
    
    9 RBF centers on a 3x3 grid in (x,v) phase space + 1 bias term.
    Gaussian width chosen for smooth local interpolation.
    """
    # Fixed RBF centers (3x3 grid covering training domain)
    xs = np.array([-0.4, 0.0, 0.4])
    vs = np.array([-0.1, 0.0, 0.2])
    
    # Gaussian width
    sigma = 0.35
    
    # Unpack params: 9 RBF weights + 1 bias
    w0, w1, w2, w3, w4, w5, w6, w7, w8, bias = params
    
    # Precompute squared distances to each center
    # Center 0: (xs[0], vs[0]) = (-0.4, -0.1)
    r2_0 = (x - xs[0])**2 + (v - vs[0])**2
    # Center 1: (xs[0], vs[1]) = (-0.4, 0.0)
    r2_1 = (x - xs[0])**2 + (v - vs[1])**2
    # Center 2: (xs[0], vs[2]) = (-0.4, 0.2)
    r2_2 = (x - xs[0])**2 + (v - vs[2])**2
    # Center 3: (xs[1], vs[0]) = (0.0, -0.1)
    r2_3 = (x - xs[1])**2 + (v - vs[0])**2
    # Center 4: (xs[1], vs[1]) = (0.0, 0.0)
    r2_4 = (x - xs[1])**2 + (v - vs[1])**2
    # Center 5: (xs[1], vs[2]) = (0.0, 0.2)
    r2_5 = (x - xs[1])**2 + (v - vs[2])**2
    # Center 6: (xs[2], vs[0]) = (0.4, -0.1)
    r2_6 = (x - xs[2])**2 + (v - vs[0])**2
    # Center 7: (xs[2], vs[1]) = (0.4, 0.0)
    r2_7 = (x - xs[2])**2 + (v - vs[1])**2
    # Center 8: (xs[2], vs[2]) = (0.4, 0.2)
    r2_8 = (x - xs[2])**2 + (v - vs[2])**2
    
    sigma2 = sigma * sigma
    
    # Gaussian RBF values
    phi0 = np.exp(-r2_0 / (2 * sigma2))
    phi1 = np.exp(-r2_1 / (2 * sigma2))
    phi2 = np.exp(-r2_2 / (2 * sigma2))
    phi3 = np.exp(-r2_3 / (2 * sigma2))
    phi4 = np.exp(-r2_4 / (2 * sigma2))
    phi5 = np.exp(-r2_5 / (2 * sigma2))
    phi6 = np.exp(-r2_6 / (2 * sigma2))
    phi7 = np.exp(-r2_7 / (2 * sigma2))
    phi8 = np.exp(-r2_8 / (2 * sigma2))
    
    return (
        w0 * phi0 + w1 * phi1 + w2 * phi2 +
        w3 * phi3 + w4 * phi4 + w5 * phi5 +
        w6 * phi6 + w7 * phi7 + w8 * phi8 +
        bias
    )
