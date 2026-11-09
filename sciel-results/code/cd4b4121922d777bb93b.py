"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Adaptive Non-Linear Basis Expansion.
    
    Instead of a fixed product-of-sums, we construct a diverse set of 10 non-linear basis 
    functions that cover polynomial, trigonometric, and exponential-like behaviors. 
    The 10 parameters act as weights for these basis functions.
    
    Basis functions:
    1. 1 (Constant)
    2. x
    3. y
    4. z
    5. x*y
    6. y*z
    7. z*x
    8. sin(x + y)
    9. cos(z)
    10. exp(-x) (or similar decaying/growing term, scaled to be well-behaved)
    
    This allows the model to capture:
    - Linear trends
    - Simple interactions
    - Periodic components
    - Non-linear saturation or decay
    
    The BFGS optimizer will fit the 10 weights (params) to minimize MSE.
    """
    
    # Ensure inputs are arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    # Extract parameters
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9 = params
    
    # Construct basis features
    # 1. Constant
    f0 = np.ones_like(x)
    # 2-4. Linear
    f1 = x
    f2 = y
    f3 = z
    # 5-7. Interactions
    f4 = x * y
    f5 = y * z
    f6 = z * x
    # 8. Trigonometric interaction
    # Note: sin/cos are bounded, which helps stability
    f7 = np.sin(x + y)
    # 9. Trigonometric in z
    f8 = np.cos(z)
    # 10. Exponential-like term. 
    # Using exp(-x) can be unstable if x is large, but x is in [0.5, 2], so exp(-x) is in [e^-2, e^-0.5] ~ [0.13, 0.6].
    # Alternatively, use a polynomial non-linear term like x^2 or (x-y)^2.
    # Let's try a mix: maybe x^2 is safer than exp for BFGS?
    # Let's use x^2 for robustness, but the previous quadratic failed.
    # Maybe the function is something like 1/(x+y+z) or log?
    # Let's try a rational-like approximation: 1/(1+x) is smooth.
    # Or simply x^2.
    # Let's stick to a diverse set. If quadratic failed, maybe it's not just x^2.
    # Let's use (x-y)^2 to capture relative differences.
    f9 = (x - y) ** 2
    
    # Combine linearly
    pred = (w0 * f0 + w1 * f1 + w2 * f2 + w3 * f3 + 
            w4 * f4 + w5 * f5 + w6 * f6 + 
            w7 * f7 + w8 * f8 + w9 * f9)
    
    # Safety check
    if not np.all(np.isfinite(pred)):
        # Return large constant to penalize, but keep finite
        return np.full_like(x, 1e6)
        
    return pred
# EVOLVE-BLOCK-END
