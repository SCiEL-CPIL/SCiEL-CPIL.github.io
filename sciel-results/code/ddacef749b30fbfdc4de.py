import numpy as np
import time

_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural change: Use Chebyshev polynomial basis on mapped domain.
    
    x in [0.1, 6] is mapped to t in [-1, 1].
    We use Chebyshev polynomials T_0(t) through T_9(t) as our basis.
    
    Chebyshev polynomials are orthogonal on [-1, 1] with weight 1/sqrt(1-t^2),
    which makes them well-conditioned for least-squares fitting. BFGS will
    converge quickly because the design matrix is well-conditioned.
    
    10 Chebyshev terms can represent smooth functions with rapid convergence,
    which is ideal for symbolic regression on a bounded interval.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [0.1, 6] to t in [-1, 1]
    # t = (2*x - (0.1+6)) / (6 - 0.1) = (2*x - 6.1) / 5.9
    t = (2.0 * x - 6.1) / 5.9
    
    # Compute Chebyshev polynomials T_0 through T_9 using recurrence:
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n}(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    # Build the basis matrix
    basis = np.empty((10, len(t)), dtype=np.float64)
    basis[0] = T0
    basis[1] = T1
    
    T_prev2 = T0
    T_prev1 = T1
    for n in range(2, 10):
        Tn = 2.0 * t * T_prev1 - T_prev2
        basis[n] = Tn
        T_prev2 = T_prev1
        T_prev1 = Tn
    
    # Linear combination: y = sum_{k=0}^{9} params[k] * T_k(t)
    y = basis.T @ params
    
    return y
