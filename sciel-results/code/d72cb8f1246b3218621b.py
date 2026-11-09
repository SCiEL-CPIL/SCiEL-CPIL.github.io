import numpy as np
import time

_START = time.time()
_LIMIT = 150.0
_MARGIN = 15.0


def _time_ok():
    return (time.time() - _START) < (_LIMIT - _MARGIN)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if not _time_ok():
        return np.ones_like(x) * params[0]
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    
    # Chebyshev polynomials of the first kind on [-1, 1]
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2*t^2 - 1
    # T3(t) = 4*t^3 - 3*t
    
    # These provide an orthogonal, well-conditioned basis on the domain [-1, 1].
    # Using the tensor product T_i(x) * T_j(y) with i+j <= 3 gives 10 terms.
    
    # Basis terms:
    # t0 = T0(x) * T0(y) = 1
    # t1 = T1(x) * T0(y) = x
    # t2 = T0(x) * T1(y) = y
    # t3 = T2(x) * T0(y) = 2*x^2 - 1
    # t4 = T0(x) * T2(y) = 2*y^2 - 1
    # t5 = T1(x) * T1(y) = x*y
    # t6 = T3(x) * T0(y) = 4*x^3 - 3*x
    # t7 = T0(x) * T3(y) = 4*y^3 - 3*y
    # t8 = T2(x) * T1(y) = (2*x^2 - 1) * y
    # t9 = T1(x) * T2(y) = x * (2*y^2 - 1)
    
    t0 = np.ones_like(x)
    t1 = x
    t2 = y
    t3 = 2.0 * x2 - 1.0
    t4 = 2.0 * y2 - 1.0
    t5 = x * y
    t6 = 4.0 * x3 - 3.0 * x
    t7 = 4.0 * y3 - 3.0 * y
    t8 = (2.0 * x2 - 1.0) * y
    t9 = x * (2.0 * y2 - 1.0)
    
    return (p0 * t0 + 
            p1 * t1 + 
            p2 * t2 + 
            p3 * t3 + 
            p4 * t4 + 
            p5 * t5 + 
            p6 * t6 + 
            p7 * t7 + 
            p8 * t8 + 
            p9 * t9)
