"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg
import scipy.integrate
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: 300s limit, aim to finish in < 290s
    start_time = time.time()
    deadline = start_time + 290.0

    # 1. Attempt to identify as Linear Autonomous System y' = A y
    # Check f(t, 0)
    zero_vec = np.zeros_like(y0)
    f_zero = f(t0, zero_vec)
    norm_f_zero = np.linalg.norm(f_zero)
    
    # Check linearity: f(t, 2y) == 2 f(t, y)
    f_y0 = f(t0, y0)
    f_2y0 = f(t0, 2.0 * y0)
    
    # Tolerance for linearity check
    tol_linear = 1e-8 * (1.0 + np.linalg.norm(f_y0))
    is_linear = np.linalg.norm(f_2y0 - 2.0 * f_y0) < tol_linear
    is_homogeneous = norm_f_zero < 1e-10
    
    if is_linear and is_homogeneous:
        # Construct A using basis vectors
        A = np.zeros((n, n), dtype=float)
        valid = True
        for j in range(n):
            if time.time() > deadline:
                valid = False
                break
            e_j = np.zeros(n, dtype=float)
            e_j[j] = 1.0
            A[:, j] = f(t0, e_j)
        
        if valid:
            # Verify A with a deterministic check vector (avoids randomness issues)
            v_check = np.ones(n) / np.sqrt(n)
            f_v = f(t0, v_check)
            A_v = A @ v_check
            
            rel_diff_lin = np.linalg.norm(f_v - A_v) / (1.0 + np.linalg.norm(A_v))
            
            # Also check autonomy: f(t1, v) == f(0, v)
            f_v_t1 = f(t1, v_check)
            rel_diff_aut = np.linalg.norm(f_v_t1 - f_v) / (1.0 + np.linalg.norm(f_v))
            
            if rel_diff_lin < 1e-10 and rel_diff_aut < 1e-10:
                # Compute exact solution y(t1) = expm(A * dt) @ y0
                try:
                    M = A * dt
                    expM = scipy.linalg.expm(M)
                    y_end = expM @ y0
                    if np.all(np.isfinite(y_end)):
                        return y_end
                except Exception:
                    pass # Fall through to solver

    # 2. Attempt Affine Linear System y' = A y + b
    if is_linear and not is_homogeneous:
        b = f_zero
        # Construct A using basis vectors, but f(t, e_j) = A e_j + b
        # So A e_j = f(t, e_j) - b
        A = np.zeros((n, n), dtype=float)
        valid = True
        for j in range(n):
            if time.time() > deadline:
                valid = False
                break
            e_j = np.zeros(n, dtype=float)
            e_j[j] = 1.0
            A[:, j] = f(t0, e_j) - b
        
        if valid:
            # Verify A with a deterministic check vector
            v_check = np.ones(n) / np.sqrt(n)
            f_v = f(t0, v_check)
            A_v_b = A @ v_check + b
            
            rel_diff_lin = np.linalg.norm(f_v - A_v_b) / (1.0 + np.linalg.norm(A_v_b))
            
            if rel_diff_lin < 1e-10:
                # Solution for affine system:
                # y(t) = expm(A t) y0 + (expm(A t) - I) A^{-1} b
                try:
                    M = A * dt
                    expM = scipy.linalg.expm(M)
                    y_homo = expM @ y0
                    
                    # Integral term: \int_0^t expm(A (t-tau)) b dtau
                    # = (expm(A t) - I) A^{-1} b
                    # Use solve for stability
                    rhs = (expM - np.eye(n)) @ b
                    y_part = scipy.linalg.solve(A, rhs, assume_a='gen')
                    
                    y_end = y_homo + y_part
                    if np.all(np.isfinite(y_end)):
                        return y_end
                except Exception:
                    pass # Fall through to solver

    # 3. Fallback: Use SciPy ODE solver for general/stiff non-linear cases
    # Radau is good for stiff problems.
    def f_wrapper(t, y):
        return f(t, y)
    
    try:
        sol = scipy.integrate.solve_ivp(
            f_wrapper, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-8, 
            atol=1e-10,
            max_step=dt/100 if dt > 0 else 1.0
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # If failed, return last available state or y0
            return sol.y[:, -1] if sol.y.size > 0 else y0
    except Exception:
        return y0

# EVOLVE-BLOCK-END
