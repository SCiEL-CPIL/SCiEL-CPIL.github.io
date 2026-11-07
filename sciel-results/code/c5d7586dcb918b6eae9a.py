import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x is in [-3, 3]
    # Approach: Fourier Series Approximation
    # This is a different family from polynomial/Chebyshev series.
    # Fourier series are orthogonal on [-L, L] with period 2L.
    # Here L=3, so period 6. Fundamental frequency w = pi/3.
    # We use 10 parameters: 1 constant, 4 (cos, sin) pairs, 1 cosine.
    
    w = np.pi / 3.0
    
    y = np.zeros_like(x, dtype=float)
    
    # Constant term
    y += params[0]
    
    # k=1
    y += params[1] * np.cos(w * x) + params[2] * np.sin(w * x)
    # k=2
    y += params[3] * np.cos(2 * w * x) + params[4] * np.sin(2 * w * x)
    # k=3
    y += params[5] * np.cos(3 * w * x) + params[6] * np.sin(3 * w * x)
    # k=4
    y += params[7] * np.cos(4 * w * x) + params[8] * np.sin(4 * w * x)
    # k=5 (Cosine only, to utilize the 10th parameter)
    y += params[9] * np.cos(5 * w * x)
    
    return y
