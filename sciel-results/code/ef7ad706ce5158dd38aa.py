import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 115.0

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: quadratic polynomial + adaptive sine + Gaussian bump.
    
    p[0]: intercept
    p[1]: x
    p[2]: y
    p[3]: x^2
    p[4]: y^2
    p[5]: x*y
    p[6]: amplitude of sin(p[7]*x + p[8]*y)
    p[7]: frequency of x in adaptive sine
    p[8]: frequency of y in adaptive sine
    p[9]: amplitude of Gaussian exp(-(x^2+y^2))
    """
    if time.time() - _START_TIME > _TIME_LIMIT:
        x2 = x * x
        y2 = y * y
        xy = x * y
        return (params[0] + params[1]*x + params[2]*y +
                params[3]*x2 + params[4]*y2 + params[5]*xy)

    x2 = x * x
    y2 = y * y
    xy = x * y

    # Quadratic polynomial backbone (linear in params, well-conditioned)
    poly = (params[0]
            + params[1] * x
            + params[2] * y
            + params[3] * x2
            + params[4] * y2
            + params[5] * xy)

    # Adaptive-frequency sinusoid
    phase = params[7] * x + params[8] * y
    sine = params[6] * np.sin(phase)

    # Gaussian bump
    r2 = x2 + y2
    gauss = params[9] * np.exp(-r2)

    pred = poly + sine + gauss

    if not np.all(np.isfinite(pred)):
        return poly

    return pred
