import time
import numpy as np
from scipy.special import roots_genlaguerre

_START = time.time()
_TIMEOUT = 630.0
_MARGIN = 30.0

def _time_left():
    return _TIMEOUT - _MARGIN - (time.time() - _START)

def construct():
    if _time_left() <= 0:
        return [4.0 * (i + 1) for i in range(12)]
    
    # Generalized Laguerre roots L_12^(11)(x), scaled to max ~ 48000
    roots = roots_genlaguerre(12, 11)
    scale = 48000.0 / max(roots)
    scaled_roots = roots * scale
    return list(scaled_roots)
