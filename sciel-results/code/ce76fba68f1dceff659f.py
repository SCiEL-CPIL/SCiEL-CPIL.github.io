import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a polynomial expansion with higher order in strain (degree 4) 
    and first order in temperature, assuming temperature acts as a 
    linear modifier of the stress-strain curve.
    Inputs are scaled to [0, 1] to anchor the expansion at the start of the data (yield point).
    """
    s_min = -0.001911
    s_max = 0.9636
    t_min = 0.06667
    t_max = 1.0
    
    s_w = s_max - s_min
    t_w = t_max - t_min
    
    s_n = (strain - s_min) / s_w
    t_n = (temp - t_min) / t_w
    
    # Precompute powers of s_n
    s2 = s_n * s_n
    s3 = s2 * s_n
    s4 = s2 * s2
    
    # Basis terms:
    # 0: 1
    # 1: s
    # 2: s^2
    # 3: s^3
    # 4: s^4
    # 5: t
    # 6: s*t
    # 7: s^2*t
    # 8: s^3*t
    # 9: s^4*t
    
    return (params[0] * 1.0 +
            params[1] * s_n +
            params[2] * s2 +
            params[3] * s3 +
            params[4] * s4 +
            params[5] * t_n +
            params[6] * s_n * t_n +
            params[7] * s2 * t_n +
            params[8] * s3 * t_n +
            params[9] * s4 * t_n)
