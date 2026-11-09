"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_cheby1


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a product of Chebyshev polynomials basis.
    
    The basis is T_i(x) * T_j(y) for i+j <= 4, giving 15 terms.
    We use 10 parameters by selecting the most important 10 terms:
    T0*T0, T1*T0, T0*T1, T2*T0, T1*T1, T0*T2, T3*T0, T2*T1, T1*T2, T0*T3
    
    This creates a well-conditioned design matrix due to Chebyshev orthogonality,
    allowing BFGS to efficiently fit smooth functions.
    """
    n = len(x)
    
    # Map x, y from [-1, 1] to [-1, 1] (already in range)
    # Compute Chebyshev polynomials T_0 through T_3 for x and y
    T0_x = np.ones(n)
    T1_x = x.copy()
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    T0_y = np.ones(n)
    T1_y = y.copy()
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Build 10 basis functions: products T_i(x)*T_j(y) with i+j <= 3, 
    # plus some higher-order terms for flexibility
    # Order: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), (2,1), (1,2), (0,3)
    basis = np.column_stack([
        T0_x * T0_y,  # 1
        T1_x * T0_y,  # x
        T0_x * T1_y,  # y
        T2_x * T0_y,  # 2x^2 - 1
        T1_x * T1_y,  # x*y
        T0_x * T2_y,  # 2y^2 - 1
        T3_x * T0_y,  # 4x^3 - 3x
        T2_x * T1_y,  # (2x^2-1)*y
        T1_x * T2_y,  # x*(2y^2-1)
        T0_x * T3_y,  # 4y^3 - 3y
    ])
    
    return basis @ params
# EVOLVE-BLOCK-END
