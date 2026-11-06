"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    # Explicit deadline guard: evaluator kills at 330s; keep a safety margin.
    deadline = time.time() + 310.0
    
    t0, t1 = t_span
    h_total = t1 - t0
    y0_arr = np.array(y0, dtype=float)
    n = len(y0_arr)
    
    # We will use a stiff solver approach. Since this is a linear system y' = A y,
    # we can extract A from f and use matrix exponentiation or a specialized solver.
    # However, the interface is generic: f(t, y).
    # For a linear system, f(t, y) = A @ y (possibly + b, but likely homogeneous).
    # Let's first check if f is linear and time-independent by probing.
    
    # Probe to see if f is of the form A @ y
    # f(0, 0) should be 0 if homogeneous linear.
    # f(0, e_i) should be column i of A.
    
    # Let's try to extract A.
    # But wait, the task says "fixed 15-dimensional stiff linear initial-value problem".
    # This implies y' = A y.
    # We can use scipy.integrate.solve_ivp with method='BDF' or 'Radau' which are good for stiff problems.
    # Or, since it's linear, we can compute expm(A * (t1-t0)) @ y0.
    # But we don't have A directly. We have to estimate it from f.
    
    # Let's estimate A.
    # A @ e_i = f(0, e_i)
    # A is n x n.
    # We need n evaluations to get the columns of A.
    # Then we can use matrix exponential.
    
    # However, f might be defined as f(t, y) = A @ y.
    # Let's check linearity.
    
    # Strategy:
    # 1. Estimate A by evaluating f(0, e_i) for i=0..n-1.
    # 2. Verify linearity: f(0, y) == A @ y for a random y.
    # 3. If linear, compute y_end = expm(A * h_total) @ y0.
    # 4. If not linear (unlikely given the task description), fall back to a robust ODE solver.
    
    # Step 1: Estimate A
    A = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        A[:, i] = f(0.0, e_i)
    
    # Step 2: Verify linearity
    # Check f(0, y0) vs A @ y0
    f_y0 = f(0.0, y0_arr)
    A_y0 = A @ y0_arr
    if not np.allclose(f_y0, A_y0, rtol=1e-5, atol=1e-8):
        # Not linear? Or maybe time-dependent?
        # Check time dependence: f(t1/2, y0) vs A @ y0
        f_mid = f(h_total / 2, y0_arr)
        if not np.allclose(f_mid, A_y0, rtol=1e-5, atol=1e-8):
            # Time-dependent or nonlinear. Fall back to solve_ivp.
            from scipy.integrate import solve_ivp
            sol = solve_ivp(f, t_span, y0_arr, method='BDF', rtol=1e-10, atol=1e-12, max_step=0.01)
            if sol.success:
                return sol.y[:, -1]
            else:
                # Try Radau
                sol = solve_ivp(f, t_span, y0_arr, method='Radau', rtol=1e-10, atol=1e-12, max_step=0.01)
                return sol.y[:, -1]
    
    # Also check if A is time-independent by checking f(t, y) for another t
    # If f is linear and time-independent, f(t, y) = A @ y for all t.
    # Let's assume it is, given the task description.
    
    # Step 3: Compute y_end = expm(A * h_total) @ y0
    from scipy.linalg import expm
    M = expm(A * h_total)
    y_end = M @ y0_arr
    
    return y_end
# EVOLVE-BLOCK-END
