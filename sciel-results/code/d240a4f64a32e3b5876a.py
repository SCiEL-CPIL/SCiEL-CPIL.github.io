import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Extract params
    p = params
    
    # Compute powers
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    
    # Scaled features
    # Target magnitude ~ 0.1
    # x ~ 0.5 -> 2x ~ 1
    # v ~ 0.3 -> 3v ~ 1
    # x3 ~ 0.125 -> 8x3 ~ 1
    # xv ~ 0.15 -> 6xv ~ 1
    # v3 ~ 0.02 -> 50v3 ~ 1
    # x2v ~ 0.05 -> 20x2v ~ 1
    # v2 ~ 0.09 -> 10v2 ~ 1
    # x5 ~ 0.03 -> 30x5 ~ 1
    # 1 ~ 1
    # x2 ~ 0.25 -> 4x2 ~ 1
    
    f0 = 2.0 * x
    f1 = 3.0 * v
    f2 = 8.0 * x3
    f3 = 6.0 * xv
    f4 = 50.0 * v3
    f5 = 20.0 * x2v
    f6 = 10.0 * v2
    f7 = 30.0 * x5
    f8 = 1.0
    f9 = 4.0 * x2
    
    return p[0]*f0 + p[1]*f1 + p[2]*f2 + p[3]*f3 + p[4]*f4 + p[5]*f5 + p[6]*f6 + p[7]*f7 + p[8]*f8 + p[9]*f9
