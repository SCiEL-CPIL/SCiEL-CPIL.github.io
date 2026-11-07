import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    u = strain
    v = temp
    
    # Polynomial surface
    # Term 1: Linear in u
    t1 = p[0] * (1.0 + p[1] * v) * u
    # Term 2: Quadratic in u
    t2 = p[2] * (1.0 + p[3] * v) * u**2
    # Term 3: Cubic in u
    t3 = p[4] * (1.0 + p[5] * v) * u**3
    # Term 4: Quartic in u
    t4 = p[6] * (1.0 + p[7] * v) * u**4
    # Intercept
    t0 = p[8] + p[9] * v
    
    stress = t0 + t1 + t2 + t3 + t4
    
    return stress
