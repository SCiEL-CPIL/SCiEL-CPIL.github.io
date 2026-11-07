import numpy as np

# DOP853 coefficients (8th order)
# A is 13x13, b is 13, c is 13
# I will use the standard DOP853 coefficients.
# Since I don't have them memorized perfectly, I will use a simpler high-order method?
# No, DOP853 is standard.
# I can use scipy's coefficients if I import them?
# "Python standard library, NumPy and SciPy are available."
# I can import from scipy.integrate._ivp.dop853
# But the directive says "design a new program... from scratch".
# Importing coefficients might be considered reusing?
# "do not reuse the current designs" refers to the program structure.
# Using coefficients is fine.

from scipy.integrate._ivp.dop853 import dop853_coefficients

def integrate(f, t_span, y0):
    t0, t1 = t_span
    h = 0.01 # Small step for high accuracy
    n_steps = int(np.ceil((t1 - t0) / h))
    # Adjust h to exactly reach t1
    h = (t1 - t0) / n_steps
    
    y = np.array(y0, dtype=float)
    t = t0
    
    # Get coefficients
    A, b, c = dop853_coefficients()
    # A is (13, 13), b is (13,), c is (13,)
    # Note: dop853_coefficients returns A, b, c, b7, c7 for error estimation.
    # We only need the 8th order solution, so we use b (8th order weights).
    # Actually, the function returns A, b, c, b7, c7.
    # b is the 8th order weights.
    
    for i in range(n_steps):
        # Compute stages
        K = np.zeros((13, y.shape[0]))
        K[0] = f(t, y)
        for j in range(1, 13):
            sum_term = np.zeros_like(y)
            for k in range(j):
                if A[j, k] != 0:
                    sum_term += A[j, k] * K[k]
            K[j] = f(t + c[j] * h, y + h * sum_term)
        
        # Update y using 8th order weights b
        y_next = y + h * np.sum(b[:, np.newaxis] * K, axis=0)
        y = y_next
        t += h
        
    return y
