"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Linear-Fourier basis.
    Uses 2 parameters for a linear trend (1, x) and 8 parameters for
    low-frequency trigonometric components (cos/sin pairs up to k=4).
    This handles both non-periodic trends and oscillatory behavior well.
    """
    # Scale argument for trigonometric functions
    # We want period 6 (from -3 to 3 is one full period for k=1)
    # k=1: cos(pi*x/3), sin(pi*x/3) -> Period 6
    # k=2: cos(2*pi*x/3), sin(2*pi*x/3) -> Period 3
    # k=3: cos(3*pi*x/3), sin(3*pi*x/3) -> Period 2
    # k=4: cos(4*pi*x/3), sin(4*pi*x/3) -> Period 1.5
    
    arg = np.pi * x / 3.0
    
    # Linear part
    y = params[0] * np.ones_like(x) + params[1] * x
    
    # Fourier part
    # k=1
    y += params[2] * np.cos(arg)
    y += params[3] * np.sin(arg)
    
    # k=2
    y += params[4] * np.cos(2.0 * arg)
    y += params[5] * np.sin(2.0 * arg)
    
    # k=3
    y += params[6] * np.cos(3.0 * arg)
    y += params[7] * np.sin(3.0 * arg)
    
    # k=4
    y += params[8] * np.cos(4.0 * arg)
    y += params[9] * np.sin(4.0 * arg)
    
    return y
# EVOLVE-BLOCK-END
