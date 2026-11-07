"""Bootstrap portfolio: Chebyshev polynomial basis for better conditioning."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Chebyshev polynomial basis.
    This basis is orthogonal on [-1, 1] and better conditioned than monomials,
    which helps the BFGS optimizer converge reliably.
    """
    # Calculate Chebyshev polynomials up to degree 3 for x and y
    # T0 = 1
    # T1 = x
    # T2 = 2x^2 - 1
    # T3 = 4x^3 - 3x
    
    t0_x = np.ones_like(x)
    t1_x = x
    t2_x = 2 * x * x - 1
    t3_x = 4 * x * x * x - 3 * x
    
    t0_y = np.ones_like(y)
    t1_y = y
    t2_y = 2 * y * y - 1
    t3_y = 4 * y * y * y - 3 * y
    
    p = params
    
    # Construct the 10 basis functions
    # 0: T0(x)T0(y) = 1
    # 1: T1(x)T0(y) = x
    # 2: T0(x)T1(y) = y
    # 3: T2(x)T0(y) = 2x^2 - 1
    # 4: T0(x)T2(y) = 2y^2 - 1
    # 5: T1(x)T1(y) = xy
    # 6: T3(x)T0(y) = 4x^3 - 3x
    # 7: T0(x)T3(y) = 4y^3 - 3y
    # 8: T2(x)T1(y) = (2x^2 - 1)y
    # 9: T1(x)T2(y) = x(2y^2 - 1)
    
    result = (
        p[0] * t0_x * t0_y +
        p[1] * t1_x * t0_y +
        p[2] * t0_x * t1_y +
        p[3] * t2_x * t0_y +
        p[4] * t0_x * t2_y +
        p[5] * t1_x * t1_y +
        p[6] * t3_x * t0_y +
        p[7] * t0_x * t3_y +
        p[8] * t2_x * t1_y +
        p[9] * t1_x * t2_y
    )
    
    return result
# EVOLVE-BLOCK-END
