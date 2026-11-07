import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: High-Degree Chebyshev Polynomial Approximation
    
    1. Map input x from [0.1, 6] to t in [-1, 1].
       t = (x - center) / scale
       center = (0.1 + 6.0) / 2 = 3.05
       scale = (6.0 - 0.1) / 2 = 2.95
       
    2. Construct Chebyshev polynomials of the first kind T_k(t) for k=0..9.
       T_0(t) = 1
       T_1(t) = t
       T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
       
    3. The model is y = sum_{k=0}^{9} c_k * T_k(t).
       This is a linear model in parameters c_k.
       BFGS will effectively perform linear regression, finding the exact
       least-squares solution for this basis.
       
    4. Why this is better:
       - Chebyshev polynomials are orthogonal on [-1, 1], leading to a well-conditioned
         design matrix. This ensures BFGS (or any gradient-based method) converges
         reliably to the global optimum for this specific function space.
       - They provide the best uniform approximation (minimax) among polynomials of the
         same degree, which is often superior to RBFs or Fourier series for smooth,
         non-periodic functions on a bounded interval.
       - Degree 9 allows capturing significant curvature and variation without the
         numerical instability of monomials x^9.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64).flatten()
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    else:
        p = p[:10]

    # Map x in [0.1, 6] to t in [-1, 1]
    # This standardization helps with numerical stability and aligns with Chebyshev orthogonality
    center = 3.05
    scale = 2.95
    t = (x - center) / scale
    
    # Clip t to [-1, 1] to handle any slight numerical overshoot or out-of-bound inputs
    # though inputs are guaranteed to be in [0.1, 6]
    t = np.clip(t, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0(t) ... T_9(t)
    # T_0 = 1
    # T_1 = t
    # T_k = 2*t*T_{k-1} - T_{k-2}
    
    T = np.zeros((10, len(x)), dtype=np.float64)
    T[0, :] = 1.0
    if len(x) > 0:
        T[1, :] = t
        for k in range(2, 10):
            T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
            
    # The prediction is the dot product of params and the Chebyshev basis
    # y = sum_{k=0}^{9} p[k] * T_k(t)
    y = T @ p
    
    return y
