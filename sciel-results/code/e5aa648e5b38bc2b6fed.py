import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    
    # 1. Check for offset (affine system y' = Ay + b)
    # f(t, 0) should be 0 for homogeneous heat equation
    f_zero = f(t0, np.zeros(n))
    
    # 2. Extract columns of A
    # A = [ f(t0, e0) ... f(t0, e_{n-1}) ]
    # If there is an offset b, f(t, y) = A y + b.
    # f(t, e_i) = A e_i + b = col_i(A) + b.
    # So col_i(A) = f(t, e_i) - b.
    
    A = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        fi = f(t0, ei)
        A[:, i] = fi - f_zero
        
    # 3. Check if A is constant in time
    # Sample a later time
    t_mid = t0 + dt / 2
    # Check one column, say col 0
    e0 = np.zeros(n)
    e0[0] = 1.0
    f_mid_0 = f(t_mid, e0)
    A_mid_col0 = f_mid_0 - f_zero # Assuming b is constant too? Or b(t)?
    # If heat equation, b=0.
    
    # Check difference
    diff = np.linalg.norm(A[:, 0] - A_mid_col0)
    
    if diff < 1e-10 * max(1.0, np.linalg.norm(A[:, 0])):
        # System is linear and time-invariant
        # Solution: y(t1) = exp(A * dt) * y0 + integral of exp(A(t-t')) b(t') dt'
        # If b=0 (f_zero is close to 0), it's just exp(A dt) y0.
        
        if np.linalg.norm(f_zero) < 1e-12:
            # Homogeneous
            exp_A_dt = scipy.linalg.expm(A * dt)
            y_final = exp_A_dt @ y0
        else:
            # Affine: y' = Ay + b
            # Solution involves integrating exp(A(t-t'))b. 
            # For constant b, y(t) = exp(A t) y0 + (exp(A t) - I) A^-1 b
            # Or use scipy.integrate.quad on the matrix? 
            # Simpler: Since b is constant, we can augment the system.
            # [y; 1]' = [A b; 0 0] [y; 1]
            A_aug = np.zeros((n+1, n+1))
            A_aug[:n, :n] = A
            A_aug[:n, n] = f_zero
            # Last row is 0
            y_aug = np.zeros(n+1)
            y_aug[:n] = y0
            y_aug[n] = 1.0
            
            exp_A_aug_dt = scipy.linalg.expm(A_aug * dt)
            y_aug_final = exp_A_aug_dt @ y_aug
            y_final = y_aug_final[:n]
            
        return y_final
    else:
        # Time-varying or non-linear fallback
        # Use a robust implicit solver
        # Since we already used ~n+2 calls, we have budget.
        # Use BDF with high precision
        from scipy.integrate import solve_ivp
        
        def f_wrapper(t, y):
            return f(t, y)
            
        sol = solve_ivp(f_wrapper, t_span, y0, method='BDF', 
                        rtol=1e-12, atol=1e-14, max_step=dt/100)
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to RK4 if BDF fails (unlikely)
            return _rk4(f, t0, t1, y0, 1000)

def _rk4(f, t0, t1, y0, steps):
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
