import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 10-term hybrid basis: polynomial (up to degree 2) + non-linear smooth terms
    # Terms: 1, x, y, x^2, y^2, xy, x^3, y^3, tanh(x*y), exp(-(x^2+y^2))
    t0 = np.ones_like(x)
    t1 = x
    t2 = y
    t3 = x ** 2
    t4 = y ** 2
    t5 = x * y
    t6 = x ** 3
    t7 = y ** 3
    t8 = np.tanh(x * y)
    t9 = np.exp(-(x ** 2 + y ** 2))
    
    return (params[0] * t0 + params[1] * t1 + params[2] * t2 +
            params[3] * t3 + params[4] * t4 + params[5] * t5 +
            params[6] * t6 + params[7] * t7 + params[8] * t8 +
            params[9] * t9)
