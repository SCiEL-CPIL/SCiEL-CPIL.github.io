import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 4.998] -> [0, 1]
    # s: [1, 4.999] -> [0, 1]
    # temp: [1, 39.98] -> [0, 1]
    # pH: [2, 12] -> [0, 1]
    b_norm = (b - 1.0) / 3.998
    s_norm = (s - 1.0) / 3.999
    t_norm = (temp - 1.0) / 38.98
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # 4th-order multivariate polynomial with 10 parameters
    # Structure:
    # p[0]: intercept
    # p[1]: linear b
    # p[2]: linear s
    # p[3]: linear temp
    # p[4]: linear pH
    # p[5]: quadratic b*s (key interaction)
    # p[6]: quadratic temp^2 (captures bell shape)
    # p[7]: quadratic pH^2 (captures bell shape)
    # p[8]: cubic temp^3 (refines bell shape)
    # p[9]: cubic b*s*temp (three-way interaction)
    
    y = (p[0]
         + p[1] * b_norm
         + p[2] * s_norm
         + p[3] * t_norm
         + p[4] * ph_norm
         + p[5] * b_norm * s_norm
         + p[6] * t_norm**2
         + p[7] * ph_norm**2
         + p[8] * t_norm**3
         + p[9] * b_norm * s_norm * t_norm)
    
    return y
