import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using a rational function with polynomial numerator and denominator.
    
    y(x) = (a0 + a1*x + a2*x^2 + a3*x^3) / (1 + b1*x + b2*x^2 + b3*x^3)
    
    Parameters (length 10):
    - a0, a1, a2, a3: numerator coefficients (4 params)
    - b1, b2, b3: denominator coefficients (3 params)
    - Extra 3 params are used for scaling/regularization to improve BFGS stability
    
    The rational form can capture:
    - Linear trends (when denominator ≈ 1)
    - Hyperbolic/asymptotic behavior
    - Poles and zeros
    - S-shaped curves
    
    To ensure numerical stability:
    - Denominator is clipped to avoid division by zero
    - Extra params provide additional degrees of freedom for the optimizer
    """
    x = np.asarray(x, dtype=float)
    
    # Extract parameters
    a0, a1, a2, a3, b1, b2, b3, s1, s2, s3 = params
    
    # Compute numerator: a0 + a1*x + a2*x^2 + a3*x^3
    num = a0 + a1 * x + a2 * x**2 + a3 * x**3
    
    # Compute denominator: 1 + b1*x + b2*x^2 + b3*x^3
    den = 1.0 + b1 * x + b2 * x**2 + b3 * x**3
    
    # Clip denominator to avoid division by zero and ensure stability
    # Use a small epsilon to prevent numerical issues
    den_clipped = np.where(np.abs(den) < 1e-8, np.sign(den) * 1e-8 + np.where(den < 0, -1e-8, 1e-8), den)
    
    # Compute the rational function
    y = num / den_clipped
    
    # Apply scaling factors from extra params for additional flexibility
    # These help BFGS navigate the parameter space more effectively
    y = y * (1.0 + 0.1 * s1 * np.tanh(s2)) + 0.01 * s3
    
    return y
