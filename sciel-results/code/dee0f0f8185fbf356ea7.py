import numpy as np
from numpy.polynomial import laguerre
import time

def construct():
    start_time = time.time()
    deadline = 600.0  # safety margin before 630s kill
    
    try:
        m = 12
        alpha = 1
        L = laguerre.Laguerre.basis(m, alpha)
        roots = L.roots()
        positive_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
        positive_roots.sort()
        
        if len(positive_roots) == 12:
            return positive_roots
        else:
            # Fallback: evenly spaced
            return [4.0 * (i + 1) for i in range(12)]
    except Exception:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return [4.0 * (i + 1) for i in range(12)]
        return [4.0 * (i + 1) for i in range(12)]
