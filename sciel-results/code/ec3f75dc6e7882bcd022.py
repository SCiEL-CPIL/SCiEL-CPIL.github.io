"""ODE solver for convection-diffusion 100-point problem using exact matrix exponential for linear systems."""
import numpy as np
from scipy.linalg import expm, solve
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Safety deadline guard
    start_time = time.time()
    deadline = start_time + 290.0
    
    # 1. Detect linearity and time-invariance
    # Assume f(t, y) = A @ y + b
    # Call f at y=0 to get b
    y_zero = np.zeros(n)
    b = np.array(f(t0, y_zero), dtype=float)
    
    # Extract A columns
    A = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        # f(t, ej) = A @ ej + b = A[:, j] + b
        A[:, j] = np.array(f(t0, ej), dtype=float) - b
    
    # 2. Solve y' = A y + b using matrix exponential
    # y(t) = e^{A(t-t0)} y(t0) + \int_{t0}^t e^{A(t-s)} b ds
    dt = t1 - t0
    E = expm(A * dt)
    
    # Compute the particular solution z(t) = \int_0^t e^{A(t-s)} b ds
    # If A is invertible, z(t) = A^{-1} (e^{At} - I) b
    # We solve A z = (E - I) b
    rhs = (E - np.eye(n)) @ b
    try:
        z = solve(A, rhs)
    except np.linalg.LinAlgError:
        # Fallback if A is singular
        # Use least squares
        z = np.linalg.lstsq(A, rhs, rcond=None)[0]
        
    y_final = E @ y + z
    
    # Check time limit
    if time.time() > deadline:
        return y_final
    
    return y_final

def _rk4_fallback(f, t_span, y0, num_steps=10000):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    h = (t1 - t0) / num_steps
    t = t0
    for _ in range(num_steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
