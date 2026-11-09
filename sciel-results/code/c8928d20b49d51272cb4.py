import numpy as np
import time
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.linalg import solve

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin
    
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    t0, t1 = t_span
    t_end = t1
    dt = t_end - t0
    
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
    # For a linear system y' = Ay + b, the exact solution is:
    # y(t) = exp(A*t) y0 + (integral from 0 to t of exp(A*(t-s)) ds) b
    #      = exp(A*t) y0 + A^(-1) (exp(A*t) - I) b  (if A is invertible)
    #      = exp(A*t) y0 + integral_0^t exp(A*tau) dtau * b
    
    # We can compute this using block matrix exponentiation or by solving for the particular solution.
    # Method: Solve (I - exp(A*dt)) y_p = -b * dt? No.
    # Particular solution for constant b: y_p = -A^(-1) b (if A invertible).
    # Then y(t) = exp(A*t) (y0 - A^(-1)b) + A^(-1)b.
    # If A is singular, use block matrix:
    # d/dt [y; 1] = [A b; 0 0] [y; 1]
    # [y(t); 1] = exp([A b; 0 0] * t) [y0; 1]
    
    # Let's use the block matrix method for robustness, but it's (N+1)x(N+1).
    # Or, we can use solve_ivp with very tight tolerances which is what the current code does.
    # The current code uses solve_ivp with Radau. This is good for stiff systems.
    # However, for linear systems, we can be more precise.
    
    # Let's try to compute expm(A*dt) directly if possible, and handle b separately.
    # y(t) = exp(A*t) y0 + \int_0^t exp(A*(t-s)) b ds
    # Let \Phi(t) = \int_0^t exp(A*tau) dtau.
    # Then \Phi(t) satisfies \Phi'(t) = A \Phi(t) + I, \Phi(0) = 0.
    # Also, \Phi(t) = A^{-1} (exp(A*t) - I) if A is invertible.
    # If A is singular, we can use the block matrix approach.
    
    # Given N=100, expm is O(N^3) which is fast.
    # The main cost in the current program is solve_ivp which might take many steps.
    # If we can get the exact solution via expm, we use 0 extra f calls (after extraction).
    # This would maximize the score because calls = 101 (extraction) + 1 (check) = 102.
    # With solve_ivp, calls might be higher depending on steps.
    
    # Let's implement the exact solution using expm.
    
    try:
        E = expm(A * dt)
        
        if has_bias:
            # Use block matrix to handle bias exactly
            M = np.zeros((N+1, N+1))
            M[:N, :N] = A
            M[:N, N] = b
            # Note: The bottom row is [0 ... 0 0]
            
            # We need expm(M * dt)
            # This is (N+1)x(N+1) expm.
            EM = expm(M * dt)
            
            z0 = np.zeros(N+1)
            z0[:N] = y0
            z0[N] = 1.0
            
            z_end = EM @ z0
            y_end = z_end[:N]
        else:
            y_end = E @ y0
            
        if np.all(np.isfinite(y_end)):
            return y_end
    except Exception:
        pass

    # Fallback to solve_ivp if expm failed
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

    # Final fallback to RK4
    return _fallback_rk4(f, t_span, y0, start_time, deadline)

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
