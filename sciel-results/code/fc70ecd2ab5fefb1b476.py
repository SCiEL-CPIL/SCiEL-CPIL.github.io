"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression basis with 10 parameters using Chebyshev polynomials
    of the first kind in x and y.
    
    Chebyshev polynomials are orthogonal on [-1, 1] with respect to the weight
    1/sqrt(1-t^2), providing superior numerical conditioning compared to
    monomials. This helps BFGS converge robustly and accurately.
    
    We use a product basis: T_i(x) * T_j(y) for i in {0,1,2,3} and j in {0,1,2},
    giving exactly 10 basis functions that span a rich space of smooth functions.
    
    Basis functions:
    T0(x)T0(y), T1(x)T0(y), T2(x)T0(y), T3(x)T0(y),
    T0(x)T1(y), T1(x)T1(y), T2(x)T1(y), T3(x)T1(y),
    T0(x)T2(y), T1(x)T2(y)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Chebyshev polynomials of the first kind:
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    
    Tx0 = np.ones_like(x)
    Tx1 = x
    Tx2 = 2.0 * x * x - 1.0
    Tx3 = 4.0 * x * x * x - 3.0 * x
    
    Ty0 = np.ones_like(y)
    Ty1 = y
    Ty2 = 2.0 * y * y - 1.0
    
    # 10 basis terms: product basis T_i(x) * T_j(y)
    b1 = Tx0 * Ty0
    b2 = Tx1 * Ty0
    b3 = Tx2 * Ty0
    b4 = Tx3 * Ty0
    b5 = Tx0 * Ty1
    b6 = Tx1 * Ty1
    b7 = Tx2 * Ty1
    b8 = Tx3 * Ty1
    b9 = Tx0 * Ty2
    b10 = Tx1 * Ty2
    
    return (params[0] * b1 + 
            params[1] * b2 + 
            params[2] * b3 + 
            params[3] * b4 + 
            params[4] * b5 + 
            params[5] * b6 + 
            params[6] * b7 + 
            params[7] * b8 + 
            params[8] * b9 + 
            params[9] * b10)
# EVOLVE-BLOCK-END
