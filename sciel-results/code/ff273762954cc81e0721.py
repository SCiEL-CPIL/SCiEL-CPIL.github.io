import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a complete 2D Chebyshev basis of degree 3.
    
    This structural change replaces the ad-hoc mix of polynomials and trigonometric
    functions with an orthogonal polynomial basis. Chebyshev polynomials are
    well-conditioned and orthogonal on [-1, 1], which significantly improves
    the convergence properties of BFGS optimization for the linear coefficients.
    
    The basis consists of all products T_i(x_n) * T_j(v_n) such that i + j <= 3,
    providing exactly 10 terms for the 10 parameters.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize inputs to [-1, 1]
    # x range: [-0.5728, 0.5275]
    x_center = (-0.5728 + 0.5275) / 2.0
    x_half_range = (0.5728 - 0.5275) / 2.0
    xn = (x - x_center) / x_half_range
    
    # v range: [-0.1754, 0.2821]
    v_center = (-0.1754 + 0.2821) / 2.0
    v_half_range = (0.2821 - 0.1754) / 2.0
    vn = (v - v_center) / v_half_range

    # Compute Chebyshev polynomials T_0 to T_3 for xn and vn
    T0_x = np.ones_like(xn)
    T1_x = xn
    T2_x = 2.0 * xn**2 - 1.0
    T3_x = 4.0 * xn**3 - 3.0 * xn

    T0_v = np.ones_like(vn)
    T1_v = vn
    T2_v = 2.0 * vn**2 - 1.0
    T3_v = 4.0 * vn**3 - 3.0 * vn

    # Basis terms: T_i(x) * T_j(v) for i + j <= 3
    # Order:
    # 0: T0_x * T0_v (degree 0)
    # 1: T1_x * T0_v (degree 1)
    # 2: T0_x * T1_v (degree 1)
    # 3: T2_x * T0_v (degree 2)
    # 4: T1_x * T1_v (degree 2)
    # 5: T0_x * T2_v (degree 2)
    # 6: T3_x * T0_v (degree 3)
    # 7: T2_x * T1_v (degree 3)
    # 8: T1_x * T2_v (degree 3)
    # 9: T0_x * T3_v (degree 3)

    b0 = T0_x * T0_v
    b1 = T1_x * T0_v
    b2 = T0_x * T1_v
    b3 = T2_x * T0_v
    b4 = T1_x * T1_v
    b5 = T0_x * T2_v
    b6 = T3_x * T0_v
    b7 = T2_x * T1_v
    b8 = T1_x * T2_v
    b9 = T0_x * T3_v

    return (
        p0 * b0
        + p1 * b1
        + p2 * b2
        + p3 * b3
        + p4 * b4
        + p5 * b5
        + p6 * b6
        + p7 * b7
        + p8 * b8
        + p9 * b9
    )
