"""
Heat2D 10x10 ODE solver using matrix exponential for linear autonomous stiff systems.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 20s safety margin before 300s limit
    
    # Step 1: Estimate the linear operator A
    # f(t, y) should be A @ y for linear autonomous system
    # First, get f at y0 to check linearity and get one column-like info
    fy0 = f(t0, y0)
    fy0 = np.asarray(fy0, dtype=float)
    
    # Check if it's linear: f(t, c*y) = c*f(t, y)
    # Use a few probe points to build A via finite differences
    # For a 100-dim system, we need 100 basis vectors
    
    # Strategy: estimate A columns using central differences
    # A[:, j] ≈ (f(t, y0 + eps*e_j) - f(t, y0 - eps*e_j)) / (2*eps)
    # But that's 200 calls. Better: use forward differences from zero or from y0
    
    # Actually, for linear system: f(t, y) = A @ y
    # So A[:, j] = f(t, e_j) where e_j is the j-th standard basis vector
    # This requires n calls (one per basis vector)
    
    # But we can be smarter: use y0 as starting point
    # f(t, y0) = A @ y0 gives us A @ y0
    # For each j: f(t, y0 + delta*e_j) - f(t, y0) = delta * A[:, j]
    # So A[:, j] = (f(t, y0 + delta*e_j) - f(t, y0)) / delta
    
    # This needs n additional calls + 1 = n+1 total
    # For n=100, that's 101 calls - very efficient!
    
    # Choose delta carefully for numerical stability
    scale = np.max(np.abs(y0)) if np.max(np.abs(y0)) > 0 else 1.0
    delta = 1e-6 * scale
    
    A = np.empty((n, n), dtype=float)
    
    # We already have f(t0, y0) = A @ y0
    # For each column j:
    for j in range(n):
        if time.time() > deadline:
            # Fallback: use what we have
            break
        
        y_perturbed = y0.copy()
        y_perturbed[j] += delta
        f_perturbed = np.asarray(f(t0, y_perturbed), dtype=float)
        
        # A[:, j] = (f(t, y0 + delta*e_j) - f(t, y0)) / delta
        A[:, j] = (f_perturbed - fy0) / delta
    
    # Verify linearity: check that A @ y0 ≈ fy0
    # If not linear, fall back to RK4
    residual = np.linalg.norm(A @ y0 - fy0)
    rel_residual = residual / (np.linalg.norm(fy0) + 1e-30)
    
    if rel_residual > 1e-4:
        # Not linear (or estimation failed), fall back to RK4
        return _rk4_fallback(f, t0, t1, y0, start_time, deadline)
    
    # Step 2: Compute the matrix exponential
    # y(t1) = expm(A * dt) @ y0
    M = A * dt
    expM = expm(M)
    y_final = expM @ y0
    
    # Ensure finite result
    if not np.all(np.isfinite(y_final)):
        return _rk4_fallback(f, t0, t1, y0, start_time, deadline)
    
    return y_final


def _rk4_fallback(f, t0, t1, y0, start_time, deadline):
    """RK4 fallback if linear assumption fails."""
    y = np.array(y0, dtype=float)
    n_steps = 5000
    h = (t1 - t0) / n_steps
    t = t0
    
    for i in range(n_steps):
        if time.time() > deadline:
            break
        
        k1 = np.asarray(f(t, y), dtype=float)
        k2 = np.asarray(f(t + h/2, y + h/2*k1), dtype=float)
        k3 = np.asarray(f(t + h/2, y + h/2*k2), dtype=float)
        k4 = np.asarray(f(t + h, y + h*k3), dtype=float)
        y = y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
