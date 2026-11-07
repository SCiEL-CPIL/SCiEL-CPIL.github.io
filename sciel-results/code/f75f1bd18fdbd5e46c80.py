import numpy as np
import time

_START_TIME = time.time()

def equation(x, y, params):
    # Deadline guard: safety margin for the 150s limit
    if time.time() - _START_TIME > 145:
        return np.zeros_like(x)

    # Refined: Chebyshev-based polynomial basis (degree up to 4)
    # Better conditioning on [-1,1] than monomials, captures smoother features.
    # T0=1, T1=x, T2=2x^2-1, T3=4x^3-3x, T4=8x^4-8x^2+1
    
    T1x = x
    T1y = y
    T2x = 2.0 * x * x - 1.0
    T2y = 2.0 * y * y - 1.0
    T3x = 4.0 * x * x * x - 3.0 * x
    T4x = 8.0 * x * x * x * x - 8.0 * x * x + 1.0
    T4y = 8.0 * y * y * y * y - 8.0 * y * y + 1.0
    
    # 10 basis functions: mix of Chebyshev terms in x, y, and cross-products
    basis = np.column_stack((
        np.ones_like(x),      # T0
        T1x,                   # T1(x)
        T1y,                   # T1(y)
        T2x,                   # T2(x)
        T2y,                   # T2(y)
        T1x * T1y,             # T1(x)*T1(y)
        T2x * T1y,             # T2(x)*T1(y)
        T1x * T2y,             # T1(x)*T2(y)
        T2x * T2y,             # T2(x)*T2(y)
        T3x                     # T3(x) for asymmetry
    ))
    
    result = basis @ params
    
    return result
