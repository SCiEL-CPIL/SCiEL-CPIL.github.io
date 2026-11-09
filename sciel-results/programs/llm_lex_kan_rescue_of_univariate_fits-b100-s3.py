"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Sigmoid-Step + Cubic Polynomial.
    
    This model combines a global polynomial trend with a local, learnable
    sigmoid (logistic) transition to capture step-like or saturating behaviors
    that smooth periodic or purely polynomial bases cannot easily represent.
    
    y(x) = p0 + p1*x + p2*x^2 + p3*x^3 
           + p4 * sigmoid(p5 * (x - p6))
           + p7 * sigmoid(p8 * (x - p9))
           
    Parameters:
    p0, p1, p2, p3: Polynomial coefficients (constant to cubic)
    p4: Amplitude of Sigmoid 1
    p5: Steepness of Sigmoid 1
    p6: Center of Sigmoid 1
    p7: Amplitude of Sigmoid 2
    p8: Steepness of Sigmoid 2
    p9: Center of Sigmoid 2
    
    The sigmoid function is defined as 1 / (1 + exp(-z)).
    To ensure numerical stability, the exponent is clipped.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
    A1, k1, c1 = p[4], p[5], p[6]
    A2, k2, c2 = p[7], p[8], p[9]
    
    # Compute polynomial part
    poly = p0 + p1 * x + p2 * x**2 + p3 * x**3
    
    # Compute sigmoid parts
    # z = k * (x - c)
    z1 = k1 * (x - c1)
    z2 = k2 * (x - c2)
    
    # Clip z to avoid overflow in exp
    z1 = np.clip(z1, -50, 50)
    z2 = np.clip(z2, -50, 50)
    
    # Sigmoid: 1 / (1 + exp(-z))
    sig1 = 1.0 / (1.0 + np.exp(-z1))
    sig2 = 1.0 / (1.0 + np.exp(-z2))
    
    # Combine
    return poly + A1 * sig1 + A2 * sig2
# EVOLVE-BLOCK-END
