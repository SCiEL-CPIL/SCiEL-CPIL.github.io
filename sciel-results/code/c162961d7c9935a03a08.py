import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize x and v to improve conditioning
    # Ranges: x ~ [-0.2, 0.17], v ~ [-0.34, 0.36]
    # Centering and scaling
    x_mean = -0.015
    x_std = 0.2
    v_mean = 0.01
    v_std = 0.35
    
    xn = (x - x_mean) / x_std
    vn = (v - v_mean) / v_std
    
    # Basis functions
    # 1. Linear restoring
    b1 = xn
    # 2. Cubic restoring
    b2 = xn**3
    # 3. Linear damping
    b3 = vn
    # 4. Cubic damping
    b4 = vn**3
    # 5. Linear coupling
    b5 = xn * vn
    # 6. Time drive sin
    b6 = np.sin(t)
    # 7. Time drive cos
    b7 = np.cos(t)
    # 8. Parametric cos
    b8 = xn * np.cos(t)
    # 9. Parametric sin
    b9 = xn * np.sin(t)
    # 10. Bias
    b10 = np.ones_like(x)
    
    # Combine
    result = (
        params[0] * b1 +
        params[1] * b2 +
        params[2] * b3 +
        params[3] * b4 +
        params[4] * b5 +
        params[5] * b6 +
        params[6] * b7 +
        params[7] * b8 +
        params[8] * b9 +
        params[9] * b10
    )
    
    return result
