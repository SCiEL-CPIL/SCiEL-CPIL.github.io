import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Chebyshev Polynomials of the first kind
    # T0 = 1
    # T1 = t
    # T2 = 2t^2 - 1
    # T3 = 4t^3 - 3t
    
    # Calculate for x
    t0x = np.ones_like(x)
    t1x = x
    t2x = 2 * x**2 - 1
    t3x = 4 * x**3 - 3 * x
    
    # Calculate for y
    t0y = np.ones_like(y)
    t1y = y
    t2y = 2 * y**2 - 1
    t3y = 4 * y**3 - 3 * y
    
    # Basis functions (Tensor product structure, selected for coverage)
    # p0: Constant
    b0 = t0x * t0y
    # p1: Linear x
    b1 = t1x * t0y
    # p2: Linear y
    b2 = t0x * t1y
    # p3: Quadratic x
    b3 = t2x * t0y
    # p4: Quadratic y
    b4 = t0x * t2y
    # p5: Interaction xy
    b5 = t1x * t1y
    # p6: Mixed quadratic (x^2 y)
    b6 = t2x * t1y
    # p7: Mixed quadratic (x y^2)
    b7 = t1x * t2y
    # p8: Biquadratic (x^2 y^2)
    b8 = t2x * t2y
    # p9: Cubic x (to capture non-symmetric higher order effects)
    b9 = t3x * t0y
    
    return (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
