"""Solve 15-dim stiff linear ODE by exploiting linearity: y(t) = expm((t1-t0)*A) @ y0.
Uses high-precision matrix exponential via mpmath if time permits to maximize accuracy."""
import numpy as np
import time
from scipy.linalg import expm
import mpmath as mp

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290  # safety margin before 330s kill
    
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)

    # Extract matrix A by evaluating f at basis vectors
    # f(t, y) = A @ y
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        col = f(t0, e)
        A[:, j] = np.asarray(col, dtype=float)
        if time.time() > deadline:
            # If we run out of time extracting A, we can't solve properly.
            # But this is extremely unlikely for n=15.
            break

    # Check if we have time for high precision
    # Standard expm is very fast for n=15
    M_std = expm(dt * A)
    y_std = M_std @ y0
    
    if time.time() < deadline - 5:
        try:
            # Use mpmath for high precision matrix exponential
            # Set precision to 100 decimal places to ensure accuracy
            mp.dps = 100
            
            # Convert A to mpmath matrix
            A_mp = mp.matrix(n, n)
            for i in range(n):
                for j in range(n):
                    A_mp[i, j] = mp.mpf(A[i, j])
            
            dt_mp = mp.mpf(dt)
            
            # Compute matrix exponential
            # mp.e**X computes exp(X)
            M_mp = mp.e ** (dt_mp * A_mp)
            
            # Convert back to numpy float64
            M_high = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    M_high[i, j] = float(M_mp[i, j])
            
            y_high = M_high @ y0
            
            # If the high precision result is finite and not NaN, use it
            if np.all(np.isfinite(y_high)):
                return y_high
            else:
                return y_std
        except Exception:
            # Fallback to standard precision if mpmath fails
            return y_std
    else:
        return y_std
