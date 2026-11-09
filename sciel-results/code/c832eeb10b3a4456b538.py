import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Basis functions:
    # 1, x, y, x^2, y^2, xy, sin(x), sin(y), cos(x), cos(y)
    # This basis is bounded and covers linear, quadratic, and periodic behaviors.
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    term_1 = np.ones_like(x)
    term_2 = x
    term_3 = y
    term_4 = x**2
    term_5 = y**2
    term_6 = x * y
    term_7 = np.sin(x)
    term_8 = np.sin(y)
    term_9 = np.cos(x)
    term_10 = np.cos(y)
    
    return (params[0] * term_1 +
            params[1] * term_2 +
            params[2] * term_3 +
            params[3] * term_4 +
            params[4] * term_5 +
            params[5] * term_6 +
            params[6] * term_7 +
            params[7] * term_8 +
            params[8] * term_9 +
            params[9] * term_10)
