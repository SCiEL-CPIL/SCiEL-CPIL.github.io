import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    
    # Compute Legendre polynomials P_0(t) to P_9(t)
    # P_0(t) = 1
    # P_1(t) = t
    # P_{k+1}(t) = ((2k+1)t P_k(t) - k P_{k-1}(t)) / (k+1)
    
    basis = np.empty((10, len(x)), dtype=np.float64)
    
    p0 = np.ones_like(t)
    p1 = t.copy()
    
    basis[0] = p0
    basis[1] = p1
    
    if len(params) > 2:
        for k in range(2, 10):
            # Calculate P_k using P_{k-1} and P_{k-2}
            # Current p0 is P_{k-2}, p1 is P_{k-1}
            # We want P_k
            # Formula: P_k = ((2(k-1)+1) * t * P_{k-1} - (k-1) * P_{k-2}) / k
            # Let j = k-1. Then P_{j+1} = ((2j+1)t P_j - j P_{j-1}) / (j+1)
            # Here we are computing index k in basis array, which corresponds to polynomial order k.
            # So we need P_k.
            # Previous step calculated P_{k-1} (stored in p1) and P_{k-2} (stored in p0).
            # To get P_k, we use the recurrence with j = k-1.
            j = k - 1
            p_k = ((2 * j + 1) * t * p1 - j * p0) / (j + 1)
            basis[k] = p_k
            p0, p1 = p1, p_k
            
    y = np.dot(params, basis)
    
    return y
