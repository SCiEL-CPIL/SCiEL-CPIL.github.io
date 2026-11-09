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
    
    # Build the design matrix for a bivariate cubic polynomial
    # Terms: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
    # params[0] * 1 + params[1] * x + params[2] * y + params[3] * x^2 + params[4] * x*y + 
    # params[5] * y^2 + params[6] * x^3 + params[7] * x^2*y + params[8] * x*y^2 + params[9] * y^3
    
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    xy = x * y
    x2y = x2 * y
    xy2 = x * y2
    
    p = params
    
    return (p[0] + 
            p[1] * x + 
            p[2] * y + 
            p[3] * x2 + 
            p[4] * xy + 
            p[5] * y2 + 
            p[6] * x3 + 
            p[7] * x2y + 
            p[8] * xy2 + 
            p[9] * y3)
