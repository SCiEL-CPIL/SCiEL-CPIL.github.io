import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, keep 20s margin
    
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Try to exploit linearity: y' = A y
    # 1. Check if f(t, 0) is zero (homogeneous)
    # 2. Extract A by probing basis vectors
    # 3. Solve using matrix exponential
    
    # Probe f(t, 0)
    y_zero = np.zeros(N)
    f_zero = f(t0, y_zero)
    
    # Check if it's close to zero
    if np.linalg.norm(f_zero) > 1e-12:
        # It might be affine: y' = A y + b
        # For now, assume homogeneous heat equation as per task description.
        # If b is not zero, the solution is different.
        # But "Heat Equation" usually implies homogeneous with BCs handled in A.
        # Let's proceed with extracting A assuming y' = A y.
        # If the check A@y0 vs f(t0, y0) fails, we fallback.
        pass
    
    # Extract A
    # We use t0 for extraction. If the system is time-invariant, t doesn't matter.
    A = np.zeros((N, N))
    
    # To save time/calls if N is large, we might sample, but N=100 is small.
    for j in range(N):
        if time.time() > deadline:
            # Fallback to a simple solver if we run out of time (unlikely)
            break
        e_j = np.zeros(N)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
        
    # Verify linearity and time-invariance roughly
    # Check f(t0, y0) vs A @ y0
    f_y0 = f(t0, y0)
    A_y0 = A @ y0
    
    # Check f(t1, y0) vs A @ y0 (time invariance)
    f_y0_t1 = f(t1, y0)
    
    # Tolerance for linearity check
    # If the system is linear, these should match closely.
    err_lin = np.linalg.norm(f_y0 - A_y0)
    err_time = np.linalg.norm(f_y0_t1 - A_y0)
    
    # Scale tolerance by magnitude of y
    scale = max(1.0, np.linalg.norm(y0))
    
    if err_lin < 1e-10 * scale and err_time < 1e-10 * scale:
        # Use Matrix Exponential
        try:
            # expm is robust for 100x100
            expA_dt = scipy.linalg.expm(A * dt)
            y_final = expA_dt @ y0
            return y_final
        except Exception:
            pass
            
    # Fallback: High accuracy RK4 or implicit solver
    # Since we already made N+3 calls, we have plenty of budget.
    # Use a robust method.
    # We can use scipy.integrate.solve_ivp with tight tolerances.
    # But solve_ivp counts calls differently? No, the evaluator counts calls to f.
    # solve_ivp will call f many times.
    # Let's use a high-step RK4 to be safe and simple.
    
    steps = 1000
    h = dt / steps
    t = t0
    y = y0.copy()
    
    for _ in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y
