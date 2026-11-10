import numpy as np
import time

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 10.0

def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Elements are (a, b) representing a + b*alpha
    # Point is ((xa, xb), (ya, yb))
    
    # We remove the parabola y = x^2
    # x = xa + xb*alpha
    # x^2 = xa^2 + 2*xa*xb*alpha + xb^2*alpha^2
    #     = (xa^2 + 3*xb^2) + (2*xa*xb)*alpha
    # So ya = (xa^2 + 3*xb^2) % 7
    #    yb = (2*xa*xb) % 7
    
    removed = np.zeros(_Q2, dtype=bool)
    
    for xa in range(p):
        for xb in range(p):
            ya = (xa * xa + 3 * xb * xb) % p
            yb = (2 * xa * xb) % p
            
            # Map point to index
            # x_idx = xa * 7 + xb
            # y_idx = ya * 7 + yb
            # point_idx = x_idx * 49 + y_idx
            x_idx = xa * p + xb
            y_idx = ya * p + yb
            pi = x_idx * q + y_idx
            removed[pi] = True
            
    N = []
    for i in range(_Q2):
        if not removed[i]:
            x = i // q
            y = i % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            N.append([[xa, xb], [ya, yb]])
            
    return np.array(N, dtype=np.int64)
