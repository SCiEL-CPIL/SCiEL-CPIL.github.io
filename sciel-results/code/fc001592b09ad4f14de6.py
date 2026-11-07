# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis of degree 9.
    Maps x in [0.1, 6] to t in [-1, 1] and uses T_0(t) ... T_9(t) as basis.
    params: 10 coefficients.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [0.1, 6.0] to t in [-1, 1]
    # t = (2*x - (max + min)) / (max - min)
    # max = 6.0, min = 0.1
    # t = (2*x - 6.1) / 5.9
    t = (2.0 * x - 6.1) / 5.9
    
    # Compute Chebyshev polynomials T_0 ... T_9
    # T_0 = 1
    # T_1 = t
    # T_{n+1} = 2*t*T_n - T_{n-1}
    
    T = np.zeros((len(t), 10))
    T[:, 0] = 1.0
    T[:, 1] = t
    
    for k in range(2, 10):
        T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
        
    pred = T @ params
    return pred
# EVOLVE-BLOCK-END
