"""ODE solver: 15-dimensional stiff linear system using analytic matrix exponential."""
import numpy as np
import time
from scipy.linalg import expm


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) on [t0, t1] with initial condition y0.
    
    Strategy:
    1. Try to detect if the system is linear with constant coefficients: y' = A*y + b
    2. If so, use the analytic solution: y(t) = expm((t-t0)*A) @ y0 + integral of expm((t-t0)*A) @ b dt
    3. If not, fall back to a robust numerical method.
    
    The score is -log10(relative error) - 0.5*log10(number of f calls), so we want
    both high accuracy AND few function evaluations.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    T = t1 - t0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # Leave 40s margin before 330s kill
    
    # Step 1: Probe the system to determine if it's linear with constant coefficients
    # Evaluate b = f(t0, 0)
    y_zero = np.zeros(n)
    b = np.array(f(t0, y_zero), dtype=float)
    
    # Evaluate A columns: A[:, j] = (f(t0, e_j) - b) / 1.0
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = np.array(f(t0, e_j), dtype=float) - b
    
    # Verify linearity and constant coefficients with a test point
    # Use a random-ish test point to check consistency
    y_test = np.ones(n) * 0.5 + 0.1 * np.arange(n)
    f_test = np.array(f(t0, y_test), dtype=float)
    f_predicted = A @ y_test + b
    residual = np.linalg.norm(f_test - f_predicted)
    scale = max(1.0, np.linalg.norm(f_predicted))
    
    # Also check at a different time to verify time-independence
    t_mid = (t0 + t1) / 2
    f_mid = np.array(f(t_mid, y_test), dtype=float)
    f_mid_predicted = A @ y_test + b
    residual_time = np.linalg.norm(f_mid - f_mid_predicted)
    
    is_linear_constant = (residual < 1e-10 * scale) and (residual_time < 1e-10 * scale)
    
    if is_linear_constant:
        # Use analytic solution
        # y(t) = expm((t-t0)*A) @ y0 + integral_0^{t-t0} expm(s*A) @ b ds
        # The integral can be computed as: if b is constant, integral = (expm(T*A) - I) @ inv(A) @ b (if A invertible)
        # More robustly: use the formula for the integral of matrix exponential
        # y(T) = expm(T*A) @ y0 + (expm(T*A) - I) @ inv(A) @ b  (if A is invertible)
        # If A is singular, use a different approach
        
        # Check if A is invertible
        try:
            A_inv = np.linalg.inv(A)
            # y(T) = expm(T*A) @ y0 + (expm(T*A) - I) @ A_inv @ b
            EA = expm(T * A)
            y_final = EA @ y0 + (EA - np.eye(n)) @ A_inv @ b
        except np.linalg.LinAlgError:
            # A is singular; use numerical integration of the ODE for the particular solution
            # Or use the formula: integral_0^T expm(s*A) ds = T * expm(0) + ... 
            # Actually, for singular A, we can use: (expm(T*A) - I) @ A^+ @ b where A^+ is pseudo-inverse
            # But this only works if b is in the range of A. Let's use a more general approach.
            # Use the fact that integral_0^T expm(s*A) ds can be computed via block matrix exponential:
            # expm([[T*A, T*b], [0, 0]]) gives [[expm(T*A), integral_0^T expm(s*A) ds], [0, 1]]
            # Actually, let's use a different trick:
            # The integral I = integral_0^T expm(s*A) ds satisfies I*A + A*I = expm(T*A) - I_n
            # We can solve this Sylvester equation, but it's complex.
            # Simpler: use the block matrix exponential approach
            block = np.zeros((2*n, 2*n))
            block[:n, :n] = T * A
            block[:n, n:] = T * np.outer(np.ones(n), b)  # This isn't quite right
            # Let me use a cleaner approach:
            # y(t) = expm(t*A) @ y0 + integral_0^t expm(s*A) @ b ds
            # For the integral, if b is constant: integral_0^t expm(s*A) ds = (expm(t*A) - I) @ A^{-1} if A invertible
            # For singular A, use pseudo-inverse and check
            A_pinv = np.linalg.pinv(A)
            EA = expm(T * A)
            # Check if (EA - I) @ A_pinv @ b gives the right answer by verifying the ODE
            y_final = EA @ y0 + (EA - np.eye(n)) @ A_pinv @ b
            
            # Verify by checking if this satisfies the ODE at t=T (approximately)
            # f(T, y_final) should equal A @ y_final + b
            f_check = np.array(f(t1, y_final), dtype=float)
            f_expected = A @ y_final + b
            if np.linalg.norm(f_check - f_expected) > 1e-8 * max(1.0, np.linalg.norm(f_expected)):
                # Pseudo-inverse approach didn't work well; fall back to numerical
                pass
            else:
                return y_final
        else:
            return y_final
    
    # Fallback: use a robust numerical method
    # Use implicit methods for stiff problems
    from scipy.integrate import solve_ivp
    
    # Try Radau first (good for stiff problems)
    sol = solve_ivp(f, [t0, t1], y0, method='Radau', rtol=1e-12, atol=1e-14, max_step=0.01)
    
    if sol.success and sol.t[-1] >= t1 - 1e-15:
        return sol.y[:, -1]
    
    # Try BDF
    sol = solve_ivp(f, [t0, t1], y0, method='BDF', rtol=1e-12, atol=1e-14, max_step=0.01)
    
    if sol.success and sol.t[-1] >= t1 - 1e-15:
        return sol.y[:, -1]
    
    # Last resort: use explicit RK45 with small steps
    sol = solve_ivp(f, [t0, t1], y0, method='RK45', rtol=1e-12, atol=1e-14, max_step=0.001)
    
    if sol.success:
        return sol.y[:, -1]
    
    # Absolute fallback: manual RK4 with many steps
    steps = 10000
    h = T / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if time.time() > deadline:
            break
    
    return y
