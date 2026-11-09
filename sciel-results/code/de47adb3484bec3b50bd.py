import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Symbolic Regression: Low-Order Polynomial + Sparse Gaussian RBF.
    
    Strategy:
    The current pure Linear RBF model (score 67.3) might suffer from:
    1. Ill-conditioning due to high correlation between fixed Gaussians.
    2. Lack of a global structural prior (e.g., quadratic trends).
    
    This proposal splits the 10 parameters into:
    - 6 parameters for a 2nd-order polynomial basis: [1, x, v, x^2, xv, v^2]
    - 4 parameters for 4 fixed-center Gaussian basis functions.
    
    The polynomial part captures the global smooth structure of the oscillator dynamics
    (which often have quadratic or cubic restoring forces). The RBF part captures
    local nonlinear deviations. This hybrid is more stable and has a stronger inductive
    bias for physical systems than a pure RBF network with the same number of parameters.
    """
    
    # Unpack parameters
    # First 6: Polynomial coefficients
    c0, c1, c2, c3, c4, c5 = params[0:6]
    
    # Last 4: RBF weights
    w1, w2, w3, w4 = params[6:10]
    
    # Polynomial Basis
    # y_poly = c0 + c1*x + c2*v + c3*x^2 + c4*x*v + c5*v^2
    y_poly = c0 + c1 * x + c2 * v + c3 * (x ** 2) + c4 * (x * v) + c5 * (v ** 2)
    
    # RBF Basis
    # Fixed centers placed at the corners of the input domain to maximize coverage
    # Input ranges: x ~ [-0.57, 0.53], v ~ [-0.18, 0.28]
    # Centers:
    cx1, cy1 = -0.45, -0.15
    cx2, cy2 =  0.45, -0.15
    cx3, cy3 = -0.45,  0.15
    cx4, cy4 =  0.45,  0.15
    
    # Bandwidth
    # Chosen to be slightly wider than the local grid spacing to ensure overlap
    sigma = 0.35
    sigma2_2 = 2.0 * sigma * sigma
    
    # Calculate Gaussian terms
    # G1
    g1 = np.exp(-((x - cx1) ** 2 + (v - cy1) ** 2) / sigma2_2)
    # G2
    g2 = np.exp(-((x - cx2) ** 2 + (v - cy2) ** 2) / sigma2_2)
    # G3
    g3 = np.exp(-((x - cx3) ** 2 + (v - cy3) ** 2) / sigma2_2)
    # G4
    g4 = np.exp(-((x - cx4) ** 2 + (v - cy4) ** 2) / sigma2_2)
    
    # Combine
    y_rbf = w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4
    
    return y_poly + y_rbf
