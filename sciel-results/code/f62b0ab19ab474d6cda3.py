"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Fourier series basis with 10 sinusoidal terms.
    The evaluator fits params via BFGS minimizing training MSE.
    
    We use a Fourier basis scaled to the domain [-2, 2]:
    - Alternating sines and cosines: cos(x), sin(x), cos(2x), sin(2x), ...
    - This provides smooth, well-conditioned, orthogonal-like basis functions
    - Captures oscillatory and non-polynomial features effectively
    - Global periodicity helps with functions that have structure across the domain
    
    The 10 basis functions are:
    1. cos(x)
    2. sin(x)
    3. cos(2x)
    4. sin(2x)
    5. cos(3x)
    6. sin(3x)
    7. cos(4x)
    8. sin(4x)
    9. cos(5x)
    10. sin(5x)
    
    Note: We don't include a constant term (T0) since the Fourier series
    naturally captures the mean through the combination of terms, and we
    have exactly 10 parameters to allocate.
    """
    # Scale x to have appropriate frequency for the domain [-2, 2]
    # The natural frequency for a function on [-2, 2] is roughly pi/2 per unit
    # so that we get about 1 period across the domain for the fundamental
    # Using x directly gives about 2/(2*pi) ≈ 0.32 periods across [-2,2]
    # Let's use a scaling factor to get better coverage
    
    # Use x directly - this gives frequencies 1, 2, 3, 4, 5 rad/unit
    # Across [-2, 2], that's about 4, 8, 12, 16, 20 radians total
    # Which is about 0.64, 1.27, 1.91, 2.55, 3.18 periods
    # This should capture a good range of oscillatory behavior
    
    # Build Fourier basis
    # cos(x), sin(x), cos(2x), sin(2x), ..., cos(5x), sin(5x)
    result = np.zeros_like(x)
    
    for k in range(1, 6):
        result += params[2*(k-1)] * np.cos(k * x)
        result += params[2*(k-1)+1] * np.sin(k * x)
    
    return result
# EVOLVE-BLOCK-END
