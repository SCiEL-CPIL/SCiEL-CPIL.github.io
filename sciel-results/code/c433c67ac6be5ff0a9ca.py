"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, keep 10s margin

    # 1. Estimate the Jacobian A and bias b
    # f(t, y) = A @ y + b
    # b = f(t0, 0)
    b = f(t0, np.zeros(n))
    
    # A columns: f(t0, e_j) - b
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j) - b
        
    # Check if we are still within time limit
    if time.time() > deadline:
        # Fallback to simple RK4 if we are running out of time (unlikely for n=100)
        return _fallback_rk4(f, t_span, y0, 1000)

    # 2. Augment the system to handle the bias term b
    # System: y' = A y + b
    # Let Y = [y; 1]. Then Y' = [[A, b], [0, 0]] Y
    A_aug = np.zeros((n + 1, n + 1))
    A_aug[:n, :n] = A
    A_aug[:n, n] = b
    # Last row is already zeros
    
    # 3. Compute the state transition matrix
    M = expm(A_aug * dt)
    
    # 4. Apply to initial condition
    y0_aug = np.zeros(n + 1)
    y0_aug[:n] = y0
    y0_aug[n] = 1.0
    
    y_final_aug = M @ y0_aug
    y_final = y_final_aug[:n]
    
    # 5. Optional: Verify linearity/affineness if time permits?
    # If the problem is strictly linear, b should be 0.
    # If the problem is non-linear, this method will be inaccurate.
    # However, "Heat Equation" is linear.
    # We assume the problem description is accurate.
    
    return y_final

def _fallback_rk4(f, t_span, y0, steps):
    """Simple RK4 fallback in case expm fails or time runs out."""
    t0, t1 = t_span
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
