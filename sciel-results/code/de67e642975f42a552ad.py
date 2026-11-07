import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    # Basis functions: 10 diverse terms
    # 0: constant
    # 1: x
    # 2: y
    # 3: z
    # 4: x^2
    # 5: y^2
    # 6: z^2
    # 7: sin(x)
    # 8: sin(y)
    # 9: x*y
    
    t0 = np.ones_like(x, dtype=np.float64)
    t1 = x
    t2 = y
    t3 = z
    t4 = x * x
    t5 = y * y
    t6 = z * z
    t7 = np.sin(x)
    t8 = np.sin(y)
    t9 = x * y
    
    result = (p[0] * t0 +
              p[1] * t1 +
              p[2] * t2 +
              p[3] * t3 +
              p[4] * t4 +
              p[5] * t5 +
              p[6] * t6 +
              p[7] * t7 +
              p[8] * t8 +
              p[9] * t9)
    
    return result
