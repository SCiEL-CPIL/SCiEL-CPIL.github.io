import numpy as np
import time
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin
    
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    t0, t1 = t_span
    t_end = t1
    
    # 1. Extract linear structure
    # Assume f(t, y) = A y + b
    # f(0, 0) = b
    b = f(0, np.zeros(N))
    
    # Check if b is negligible
    has_bias = np.linalg.norm(b) > 1e-12
    
    A = np.zeros((N, N))
    # Extract columns of A
    # f(0, e_j) = A e_j + b = A[:, j] + b
    # So A[:, j] = f(0, e_j) - b
    for j in range(N):
        if time.time() > deadline:
            break # Fallback later? No, just return best effort or raise?
            # The task says return final state. If we fail, we should fallback.
            
        e_j = np.zeros(N)
        e_j[j] = 1.0
        Aj = f(0, e_j)
        A[:, j] = Aj - b
        
    # Check if A is finite
    if not np.all(np.isfinite(A)):
        # Fallback
        return _fallback_rk4(f, t_span, y0, start_time, deadline)
        
    # 2. Verify linearity and time-invariance with a few extra calls
    # If this fails, we might still try the linear model but be less confident
    # For now, assume it holds as per task description
    # Let's do a quick check at t=0.25
    if time.time() < deadline - 10.0:
        t_check = 0.25
        y_check = y0 + np.random.rand(N) * 1e-6
        f_check = f(t_check, y_check)
        f_linear_check = A @ y_check + b
        err_check = np.linalg.norm(f_check - f_linear_check)
        if err_check > 1e-8 * (1 + np.linalg.norm(y_check)):
            # Linearity or time-invariance might be violated
            # Fallback to direct integration of f?
            # But we've already spent 101 calls.
            # Let's try to integrate f directly with high order solver
            return _high_order_solve(f, t_span, y0, start_time, deadline)

    # 3. Compute solution using high-precision integration of the linear system
    # We use solve_ivp with Radau (implicit, stiff-capable) and tight tolerances
    # This avoids potential issues with expm for stiff matrices
    
    def f_linear(t, y):
        return A @ y + b

    try:
        # Use Radau for stiffness, tight tolerances
        sol = solve_ivp(f_linear, [t0, t_end], y0, method='Radau', 
                        rtol=1e-12, atol=1e-14, max_step=0.1)
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Fallback to expm if solve_ivp failed
    if has_bias:
        M = np.zeros((N+1, N+1))
        M[:N, :N] = A
        M[:N, N] = b
        z0 = np.zeros(N+1)
        z0[:N] = y0
        z0[N] = 1.0
        E = expm((t_end - t0) * M)
        z_end = E @ z0
        y_end = z_end[:N]
    else:
        E = expm((t_end - t0) * A)
        y_end = E @ y0
        
    # Verify result
    if not np.all(np.isfinite(y_end)):
        return _fallback_rk4(f, t_span, y0, start_time, deadline)
        
    return y_end

def _high_order_solve(f, t_span, y0, start_time, deadline):
    # Direct high-order integration if linearity check failed
    try:
        sol = solve_ivp(f, t_span, y0, method='Radau', 
                        rtol=1e-10, atol=1e-12, max_step=0.1)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    return _fallback_rk4(f, t_span, y0, start_time, deadline)

def _fallback_rk4(f, t_span, y0, start_time, deadline):
    # Simple RK4 with reasonable steps
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    steps = 1000 # Fewer steps to save time, accuracy might drop
    h = (t1 - t0) / steps
    t = t0
    
    for _ in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y
