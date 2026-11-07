# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Degree-3 bivariate polynomial model using Chebyshev basis.
    Chebyshev polynomials provide better conditioning than monomials,
    which helps BFGS converge reliably.
    
    Basis functions:
    T0(x)T0(y), T1(x)T0(y), T0(x)T1(y), T2(x)T0(y), T1(x)T1(y), T0(x)T2(y),
    T3(x)T0(y), T2(x)T1(y), T1(x)T2(y), T0(x)T3(y)
    
    params[0:10] are the coefficients for these terms.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has length 10
    if p.size < 10:
        p = np.concatenate([p, np.ones(10 - p.size)])
    p = p[:10]

    # Ensure x is 2D (N, 2)
    if x.ndim == 1:
        if x.size % 2 == 0:
             x = x.reshape(-1, 2)
        else:
             return np.zeros(y.shape)

    # Handle case where x might be (2, N)
    if x.shape[0] == 2 and x.shape[1] != 2:
        x = x.T

    n = x.shape[0]
    if n == 0:
        return np.zeros(0)

    x1 = x[:, 0]
    x2 = x[:, 1]

    # Clip inputs to [-1, 1] for numerical stability in Chebyshev eval
    x1 = np.clip(x1, -1.0, 1.0)
    x2 = np.clip(x2, -1.0, 1.0)

    # Compute Chebyshev polynomials up to degree 3
    # T0(z) = 1
    # T1(z) = z
    # T2(z) = 2z^2 - 1
    # T3(z) = 4z^3 - 3z
    
    T0_x = np.ones(n)
    T1_x = x1
    T2_x = 2.0 * x1 * x1 - 1.0
    T3_x = 4.0 * x1 * x1 * x1 - 3.0 * x1

    T0_y = np.ones(n)
    T1_y = x2
    T2_y = 2.0 * x2 * x2 - 1.0
    T3_y = 4.0 * x2 * x2 * x2 - 3.0 * x2

    # Basis functions (tensor product of Chebyshev polynomials)
    # Order: T0(x)T0(y), T1(x)T0(y), T0(x)T1(y), T2(x)T0(y), T1(x)T1(y), 
    #        T0(x)T2(y), T3(x)T0(y), T2(x)T1(y), T1(x)T2(y), T0(x)T3(y)
    
    term0 = T0_x * T0_y
    term1 = T1_x * T0_y
    term2 = T0_x * T1_y
    term3 = T2_x * T0_y
    term4 = T1_x * T1_y
    term5 = T0_x * T2_y
    term6 = T3_x * T0_y
    term7 = T2_x * T1_y
    term8 = T1_x * T2_y
    term9 = T0_x * T3_y

    # Combine with params
    pred = (p[0] * term0 + 
            p[1] * term1 + 
            p[2] * term2 + 
            p[3] * term3 + 
            p[4] * term4 + 
            p[5] * term5 + 
            p[6] * term6 + 
            p[7] * term7 + 
            p[8] * term8 + 
            p[9] * term9)
            
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
        
    return pred

# EVOLVE-BLOCK-END
