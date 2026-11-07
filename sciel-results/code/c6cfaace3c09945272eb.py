import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a 9th-degree Chebyshev polynomial basis.
    Maps input x in [-3, 3] to t in [-1, 1] to use standard Chebyshev polynomials.
    y = sum_{k=0}^{9} p_k * T_k(x/3)
    """
    # Map x to [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2t T_n(t) - T_{n-1}(t)
    
    # We need 10 terms (k=0 to 9)
    # T_0
    T0 = np.ones_like(t)
    # T_1
    T1 = t.copy()
    
    y = params[0] * T0 + params[1] * T1
    
    if len(params) > 2:
        T_prev = T0
        T_curr = T1
        for k in range(2, 10):
            T_next = 2.0 * t * T_curr - T_prev
            y += params[k] * T_next
            T_prev = T_curr
            T_curr = T_next
            
    return y
