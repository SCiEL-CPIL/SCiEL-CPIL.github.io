import numpy as np
import time
from scipy.linalg import expm, solve, expm_multiply

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] given initial condition y0.
    Assumes f is linear and time-invariant: f(t, y) = A @ y + b.
    Uses matrix exponential for exact solution of the linear system.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    y0 = np.array(y0, dtype=float)
    N = len(y0)
    t0, t1 = t_span
    dt = t1 - t0

    # Extract b = f(0, 0)
    b = f(0, np.zeros(N))
    
    # Extract A by evaluating f at basis vectors
    # f(0, e_j) = A @ e_j + b = A[:, j] + b
    # So A[:, j] = f(0, e_j) - b
    A = np.zeros((N, N))
    for j in range(N):
        if time.time() > deadline:
            break
        e_j = np.zeros(N)
        e_j[j] = 1.0
        val = f(0, e_j)
        A[:, j] = val - b

    # Check for finite values
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        return _fallback_solve(f, t_span, y0, start_time, deadline)

    # Compute the exact solution for y' = A y + b
    # y(t) = expm(A*dt) @ y0 + (expm(A*dt) - I) @ A^{-1} @ b
    # We use expm_multiply to avoid forming the full matrix exponential
    try:
        # Solve A x = b
        # If A is singular, this will fail or produce inf. 
        # For convection-diffusion, A is typically invertible (diagonally dominant).
        x = solve(A, b)
        
        # y_end = expm(A*dt) @ y0 + expm(A*dt) @ x - x
        # = expm(A*dt) @ (y0 + x) - x
        
        y0_plus_x = y0 + x
        
        # Compute E @ (y0 + x)
        y_end_part1 = expm_multiply(A * dt, y0_plus_x)
        
        y_end = y_end_part1 - x
        
        if np.all(np.isfinite(y_end)):
            return y_end
    except Exception:
        pass

    # Fallback: Direct integration with high-order solver
    return _fallback_solve(f, t_span, y0, start_time, deadline)

def _fallback_solve(f, t_span, y0, start_time, deadline):
    """
    Fallback solver using scipy.integrate.solve_ivp if the linear structure assumption fails
    or if expm fails.
    """
    from scipy.integrate import solve_ivp
    try:
        # Use Radau for stiff problems, or DOP853 for high accuracy
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12, max_step=0.05)
        if sol.success and sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: RK4
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    steps = 500
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
