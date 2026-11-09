"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a bivariate Chebyshev polynomial basis of total degree 3.
    
    Basis functions:
    1. T0(x)T0(y) = 1
    2. T1(x)T0(y) = x
    3. T0(x)T1(y) = y
    4. T2(x)T0(y) = 2x^2 - 1
    5. T1(x)T1(y) = xy
    6. T0(x)T2(y) = 2y^2 - 1
    7. T3(x)T0(y) = 4x^3 - 3x
    8. T2(x)T1(y) = (2x^2 - 1)y
    9. T1(x)T2(y) = x(2y^2 - 1)
    10. T0(x)T3(y) = 4y^3 - 3y
    
    Chebyshev polynomials are orthogonal on [-1, 1], providing a well-conditioned basis
    for BFGS optimization compared to monomials.
    """
    # Chebyshev polynomials of the first kind
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x * x - 1
    T3_x = 4 * x**3 - 3 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y * y - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Construct the prediction as a linear combination of basis functions
    pred = (
        params[0] * T0_x * T0_y +
        params[1] * T1_x * T0_y +
        params[2] * T0_x * T1_y +
        params[3] * T2_x * T0_y +
        params[4] * T1_x * T1_y +
        params[5] * T0_x * T2_y +
        params[6] * T3_x * T0_y +
        params[7] * T2_x * T1_y +
        params[8] * T1_x * T2_y +
        params[9] * T0_x * T3_y
    )
    
    return pred
# EVOLVE-BLOCK-END
