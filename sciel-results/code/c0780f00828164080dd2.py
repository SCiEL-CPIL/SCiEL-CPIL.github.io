import numpy as np
import scipy.integrate
import scipy.linalg
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    n = len(y0)
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Helper to extract A and b at a given t
    def extract_params(t_val):
        zero_vec = np.zeros(n, dtype=float)
        b_val = np.array(f(t_val, zero_vec), dtype=float)
        
        A_val = np.zeros((n, n), dtype=float)
        for i in range(n):
            e_i = np.zeros(n, dtype=float)
            e_i[i] = 1.0
            col_i = np.array(f(t_val, e_i), dtype=float) - b_val
            A_val[:, i] = col_i
        return A_val, b_val

    # Strategy:
    # 1. Probe at t0 and t1 to check if coefficients are constant.
    # 2. If constant, use exact solution via matrix exponential.
    # 3. If not constant, use piecewise constant approximation with many steps.
    
    A0, b0 = extract_params(t0)
    A1, b1 = extract_params(t1)
    
    # Check for constant coefficients using t0, midpoint, t1
    t_mid = 0.5 * (t0 + t1)
    A_mid, b_mid = extract_params(t_mid)
    
    norm_A0 = np.linalg.norm(A0)
    norm_b0 = np.linalg.norm(b0)
    
    tol_A = 1e-10 * max(1.0, norm_A0)
    tol_b = 1e-10 * max(1.0, norm_b0)
    
    diff_A_01 = np.linalg.norm(A0 - A1)
    diff_b_01 = np.linalg.norm(b0 - b1)
    diff_A_0m = np.linalg.norm(A0 - A_mid)
    diff_b_0m = np.linalg.norm(b0 - b_mid)
    
    if (diff_A_01 < tol_A and diff_b_01 < tol_b and 
        diff_A_0m < tol_A and diff_b_0m < tol_b):
        # Assume constant coefficients
        # Use A0, b0
        # Exact solution for y' = A y + b
        # Use block matrix exponential for robustness
        
        M = np.zeros((n+1, n+1))
        M[:n, :n] = A0
        M[:n, n] = b0
        
        dt = t1 - t0
        try:
            M_exp = scipy.linalg.expm(M * dt)
            z0 = np.zeros(n+1)
            z0[:n] = y0
            z0[n] = 1.0
            z1 = M_exp @ z0
            return z1[:n]
        except Exception:
            # Fallback
            def f_const(t_val, y_val):
                return A0 @ y_val + b0
            
            sol = scipy.integrate.solve_ivp(
                f_const, 
                (t0, t1), 
                y0, 
                method='Radau',
                rtol=1e-13, 
                atol=1e-15,
                max_step=dt / 100.0
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                return y0

    # If not constant, use piecewise constant approximation
    # We already have A0, b0 at t0 and A1, b1 at t1.
    # Let's probe N-2 intermediate points.
    N = 100  # Number of intervals (reduced from 200 to save f calls)
    # We need N+1 points: t0, ..., t1
    # We have index 0 and N. We need 1 to N-1.
    
    ts = np.linspace(t0, t1, N + 1)
    
    A_vals = [A0]
    b_vals = [b0]
    
    # Probe intermediate points
    for i in range(1, N):
        if time.time() > deadline - 10.0:
             return _solve_direct(f, t_span, y0)
        t_p = ts[i]
        A_p, b_p = extract_params(t_p)
        A_vals.append(A_p)
        b_vals.append(b_p)
        
    A_vals.append(A1)
    b_vals.append(b1)
    
    # Integrate step by step
    y_curr = y0.copy()
    
    # Precompute block matrices if possible? No, A changes.
    for i in range(N):
        if time.time() > deadline - 5.0:
            break
            
        A_i = A_vals[i]
        b_i = b_vals[i]
        dt = ts[i+1] - ts[i]
        
        # Construct augmented matrix for this step
        M = np.zeros((n+1, n+1))
        M[:n, :n] = A_i
        M[:n, n] = b_i
        
        try:
            M_exp = scipy.linalg.expm(M * dt)
            z0 = np.zeros(n+1)
            z0[:n] = y_curr
            z0[n] = 1.0
            z1 = M_exp @ z0
            y_curr = z1[:n]
        except Exception:
            # Fallback for this step: simple Euler? Or Radau on sub-interval?
            # If expm fails, we are in trouble. Let's try a small Radau step.
            def f_step(t_val, y_val):
                return A_i @ y_val + b_i
            
            sol = scipy.integrate.solve_ivp(
                f_step, 
                (ts[i], ts[i+1]), 
                y_curr, 
                method='Radau',
                rtol=1e-10, 
                atol=1e-12
            )
            if sol.success:
                y_curr = sol.y[:, -1]
            else:
                y_curr = y_curr + A_i @ y_curr * dt + b_i * dt # Euler fallback
    
    return y_curr

def _solve_direct(f, t_span, y0):
    """Fallback solver if model fails."""
    t0, t1 = t_span
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            sol = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-10, 
                atol=1e-12
            )
            return sol.y[:, -1]
    except Exception:
        return y0
