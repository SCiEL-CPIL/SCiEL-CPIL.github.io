"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Strategy:
    1. Detect if the system is linear and time-invariant: f(t, y) = A y + b.
    2. Reconstruct A and b from evaluations.
    3. Compute the exact solution y(t1) = exp(A*(t1-t0)) y0 + (A^-1 (exp(A*(t1-t0)) - I)) b 
       if A is invertible, or more generally using the matrix exponential of the augmented system or 
       simply integrating the affine term if b is constant.
       Actually, for y' = A y + b, the solution is:
       y(t) = exp(A(t-t0)) y0 + ∫_0^{t-t0} exp(A τ) dτ b
       The integral term can be computed as A^{-1} (exp(A Δt) - I) b if A is invertible.
       Or we can augment the state: z = [y; 1], z' = [[A, b], [0, 0]] z.
       Then z(t1) = exp([[A, b], [0, 0]] Δt) [y0; 1].
       The size is (n+1) x (n+1). For n=100, this is 101x101. Very fast.
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    n = len(y0_arr)
    dt_total = t1 - t0
    
    if dt_total == 0:
        return y0_arr

    # Step 1: Check for affine linear time-invariant system: f(t, y) = A y + b
    
    # Estimate b = f(t0, 0)
    y_zero = np.zeros_like(y0_arr)
    b = f(t0, y_zero)
    
    # Check time invariance with a test point
    # If f(t1, y0) - b is proportional to y0 in the same way as f(t0, y0) - b, it's likely TI.
    # Better: Reconstruct A from basis vectors at t0.
    
    # Reconstruct A
    # A @ e_j = f(t0, e_j) - b
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros_like(y0_arr)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j) - b
        
    # Verify linearity and time-invariance
    # Check 1: Does A y0 + b match f(t0, y0)?
    pred_t0 = A @ y0_arr + b
    actual_t0 = f(t0, y0_arr)
    err_t0 = np.linalg.norm(pred_t0 - actual_t0)
    
    # Check 2: Does A y0 + b match f(t1, y0)?
    actual_t1 = f(t1, y0_arr)
    err_t1 = np.linalg.norm(pred_t0 - actual_t1)
    
    scale = max(1.0, np.linalg.norm(actual_t0))
    
    if err_t0 < 1e-8 * scale and err_t1 < 1e-8 * scale:
        # System is affine linear and time-invariant: y' = A y + b
        # Use augmented matrix exponential
        # State vector z = [y; 1]
        # z' = [[A, b], [0, 0]] z
        # z(t1) = exp(M * dt) z(t0)
        
        M = np.zeros((n + 1, n + 1), dtype=float)
        M[:n, :n] = A
        M[:n, n] = b
        # Last row is zeros
        
        z0 = np.zeros(n + 1, dtype=float)
        z0[:n] = y0_arr
        z0[n] = 1.0
        
        z1 = expm(M * dt_total) @ z0
        
        return z1[:n]
    else:
        # Fallback: Use RK45 or similar
        # For robustness, use a high-order RK method
        # Since we don't know the stiffness, we use a moderate step size
        # But for the specific task, this branch shouldn't be hit.
        # We'll use a simple adaptive step RK45 implementation or fixed step RK4.
        # Given the penalty on calls, we want to be efficient.
        # If it's not linear, we are in trouble. But the task is fixed.
        
        # Fallback to RK4 with 10000 steps
        steps = 10000
        h = dt_total / steps
        t = t0
        y = y0_arr.copy()
        
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = f(t + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
            
        return y

# EVOLVE-BLOCK-END
