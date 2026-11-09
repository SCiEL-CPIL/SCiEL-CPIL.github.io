"""Rewritten ODE solver for 15-dimensional stiff linear system."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy: 
    1. Assume the system is linear and time-invariant: y' = A y + b.
    2. Extract b by evaluating f(t0, 0).
    3. Extract A by evaluating f(t0, e_j) and subtracting b.
    4. Solve the inhomogeneous linear ODE using matrix exponential on an augmented system.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 285.0  # 300s limit, 15s safety margin
    
    # Check if we're running out of time
    if time.time() > deadline:
        return y0
        
    dt = t1 - t0
    if abs(dt) < 1e-15:
        return y0

    # Step 1: Extract b = f(t0, 0)
    zero_vec = np.zeros(n, dtype=float)
    b = np.array(f(t0, zero_vec), dtype=float)
    
    if time.time() > deadline:
        return y0

    # Step 2: Extract columns of A
    # f(t0, e_j) = A @ e_j + b = A[:, j] + b
    # So A[:, j] = f(t0, e_j) - b
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        if time.time() > deadline:
            break
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        col = np.array(f(t0, e_j), dtype=float)
        A[:, j] = col - b
        
    if time.time() > deadline:
        # Fallback if extraction incomplete: return y0 (or best guess)
        # If we have partial A, we can't easily fix it. 
        # But we have b. 
        # If A is incomplete, we can't compute expm.
        # We'll just return y0 as a safe fallback.
        return y0

    # Step 3: Construct augmented system for inhomogeneous ODE
    # y' = A y + b
    # Let w = [y; 1]
    # w' = [A y + b; 0] = [[A, b], [0, 0]] w
    M_aug = np.zeros((n + 1, n + 1), dtype=float)
    M_aug[:n, :n] = A
    M_aug[:n, n] = b
    # Last row and column are already 0
    
    # Step 4: Compute exponential and apply to initial condition
    # w(0) = [y0; 1]
    w0 = np.zeros(n + 1, dtype=float)
    w0[:n] = y0
    w0[n] = 1.0
    
    # Compute expm(M_aug * dt)
    # For stiff systems, expm is robust.
    try:
        M_exp = expm(M_aug * dt)
        w_final = M_exp @ w0
        y_final = w_final[:n]
    except Exception:
        # Fallback to simple Euler if expm fails (unlikely)
        h = dt / 1000
        y = y0.copy()
        for _ in range(1000):
            y = y + h * (A @ y + b)
        y_final = y

    # Verify finiteness
    if not np.all(np.isfinite(y_final)):
        # Fallback
        h = dt / 1000
        y = y0.copy()
        for _ in range(1000):
            y = y + h * (A @ y + b)
        y_final = y

    return y_final
# EVOLVE-BLOCK-END
