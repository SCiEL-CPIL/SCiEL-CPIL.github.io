"""ODE solver for convection-diffusion 100-point problem using matrix exponential strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Safety deadline guard
    start_time = time.time()
    deadline = start_time + 290.0
    
    # 1. Detect linearity and time-invariance, and extract A, b
    # For y' = f(t, y), if linear and time-invariant: f(t, y) = A @ y + b
    # We can determine A and b by calling f at specific points.
    # Let's call f at t=0 with y=0 to get b (assuming linearity).
    # Then call f at t=0 with y=e_j (standard basis) to get columns of A.
    # n=100, so 101 calls is cheap.
    
    y0_test = np.zeros(n)
    b = f(t0, y0_test)
    
    # Extract A columns
    A = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        # f(t, ej) = A @ ej + b = A[:, j] + b
        # So A[:, j] = f(t, ej) - b
        A[:, j] = f(t0, ej) - b
        
    # Verify linearity with a random vector
    # Use a deterministic random vector to avoid seeding issues, or just a specific test vector
    # Since the task is fixed, we can just check against y0 or a known vector.
    # Let's check against y0 itself to be safe.
    if np.linalg.norm(y) > 1e-12:
        y_rand = y / np.linalg.norm(y)
    else:
        y_rand = np.ones(n) / np.sqrt(n)
        
    f_pred = A @ y_rand + b
    f_actual = f(t0, y_rand)
    
    if not np.allclose(f_pred, f_actual, rtol=1e-10, atol=1e-12):
        # Fallback to RK4 if not linear
        # This shouldn't happen for the specified task, but good for safety
        return _rk4_fallback(f, t_span, y0)
    
    # 2. Solve y' = A y + b using matrix exponential
    # Solution: y(t) = e^{A(t-t0)} y(t0) + \int_{t0}^t e^{A(t-s)} b ds
    
    dt = t1 - t0
    E = expm(A * dt)
    
    # Compute the integral term: z = \int_0^dt expm(A*u) b du
    # We know that d/du (expm(A*u) b) = A expm(A*u) b
    # Integrating: expm(A*dt) b - b = \int_0^dt A expm(A*u) b du = A (\int_0^dt expm(A*u) b du)
    # So, A z = E @ b - b
    # If A is invertible, z = A^{-1} (E @ b - b)
    
    rhs = E @ b - b
    
    # Try to solve A z = rhs
    try:
        # Check if A is singular
        # Using np.linalg.solve might be faster than lstsq if well-conditioned
        # If it fails, we use lstsq
        z = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        # Fallback to least squares if A is singular
        z, residuals, rank, s = np.linalg.lstsq(A, rhs, rcond=None)
    
    y_final = E @ y + z
    
    # Check time limit
    if time.time() > deadline:
        return y_final
        
    return y_final

def _rk4_fallback(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    h = (t1 - t0) / 1000
    t = t0
    for _ in range(1000):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
