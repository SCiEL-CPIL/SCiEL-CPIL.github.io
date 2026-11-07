import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Combine polynomial (up to cubic) and trigonometric basis functions
    # This hybrid approach covers both polynomial and oscillatory target functions
    # with 10 parameters total.
    
    p = params
    
    # Polynomial terms
    x2 = x * x
    y2 = y * y
    xy = x * y
    x3 = x2 * x
    y3 = y2 * y
    x2y = x2 * y
    xy2 = x * y2
    
    # Trigonometric terms
    sin_px = np.sin(np.pi * x)
    sin_py = np.sin(np.pi * y)
    cos_px = np.cos(np.pi * x)
    cos_py = np.cos(np.pi * y)
    
    # Hybrid basis: 4 polynomial + 4 trigonometric + 2 mixed
    # p0: constant
    # p1: x
    # p2: y
    # p3: x*y
    # p4: sin(pi*x)
    # p5: sin(pi*y)
    # p6: cos(pi*x)
    # p7: cos(pi*y)
    # p8: x^2 + y^2 (radial quadratic)
    # p9: x^2*y + x*y^2 (cubic interaction)
    
    pred = (p[0]
            + p[1] * x
            + p[2] * y
            + p[3] * xy
            + p[4] * sin_px
            + p[5] * sin_py
            + p[6] * cos_px
            + p[7] * cos_py
            + p[8] * (x2 + y2)
            + p[9] * (x2y + xy2))
    
    return pred
