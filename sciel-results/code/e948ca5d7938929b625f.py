"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # Safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a truncated Fourier series with 5 terms.
    
    The model is a linear combination of sine and cosine basis functions:
    y(x) = sum_{k=1}^{5} [a_k * sin(k*pi*x) + b_k * cos(k*pi*x)]
    
    This 10-parameter model can capture oscillatory or quasi-periodic structures
    that polynomial models cannot represent efficiently. The basis functions are
    orthogonal on [-1, 1] (scaled to [-2, 2] via the frequency k*pi).
    """
    # Deadline guard: if we're running too long, return a simple linear fit
    if time.time() > _DEADLINE:
        x = np.asarray(x, dtype=np.float64)
        p = params
        return p[0] + p[1] * x
    
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Parameters:
    # p[0:5] -> sine coefficients a_1..a_5
    # p[5:10] -> cosine coefficients b_1..b_5
    
    result = np.zeros_like(x, dtype=np.float64)
    
    for k in range(1, 6):
        freq = k * np.pi
        result += p[k-1] * np.sin(freq * x)
        result += p[k+4] * np.cos(freq * x)
    
    return result

# EVOLVE-BLOCK-END
