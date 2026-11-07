import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Compute Chebyshev basis up to degree 9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    basis = np.zeros((len(x), 10))
    basis[:, 0] = 1.0
    if len(x) > 0:
        basis[:, 1] = t
        for k in range(2, 10):
            basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
            
    return basis @ params
