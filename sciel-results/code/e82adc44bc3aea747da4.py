"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fourier series basis on domain [0.1, 6]
    # Center and scale x to [0, 2*pi] for better conditioning
    # Midpoint of [0.1, 6] is 3.05, half-width is 2.95
    # Map x from [0.1, 6] to t in [0, 2*pi]
    # t = 2*pi * (x - 0.1) / (6 - 0.1)
    t = 2 * np.pi * (x - 0.1) / 5.9
    
    # Basis functions: constant, then pairs of (cos, sin) for k=1..4
    # 1. Constant
    b0 = np.ones_like(x)
    # 2. cos(t)
    b1 = np.cos(t)
    # 3. sin(t)
    b2 = np.sin(t)
    # 4. cos(2t)
    b3 = np.cos(2 * t)
    # 5. sin(2t)
    b4 = np.sin(2 * t)
    # 6. cos(3t)
    b5 = np.cos(3 * t)
    # 7. sin(3t)
    b6 = np.sin(3 * t)
    # 8. cos(4t)
    b7 = np.cos(4 * t)
    # 9. sin(4t)
    b8 = np.sin(4 * t)
    # 10. cos(5t)
    b9 = np.cos(5 * t)
    
    # Linear combination
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 + 
            params[3] * b3 + params[4] * b4 + params[5] * b5 + 
            params[6] * b6 + params[7] * b7 + params[8] * b8 + 
            params[9] * b9)
# EVOLVE-BLOCK-END
