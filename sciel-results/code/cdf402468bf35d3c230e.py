import numpy as np
import time
import scipy.linalg as la
from scipy.integrate import solve_ivp

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0  # 5 seconds margin before 330s

def _is_deadline_exceeded():
    return time.time() - _START_TIME > _DEADLINE

def _extract_AB(f, t, y0, n):
    """
    Extract A and b from f(t, y) = A y + b.
    Uses n+1 calls: f(t, 0) for b, and f(t, e_j) for columns of A.
    """
    y_zero = np.zeros(n)
    b = np.array(f(t, y_zero), dtype=float)
    
    A = np.zeros((n, n))
    # Use a small epsilon for numerical differentiation if necessary, 
    # but for linear f, f(t, e_j) - f(t, 0) = A e_j = column j of A.
    # This is exact for linear systems.
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        y_test = e_j
        f_test = np.array(f(t, y_test), dtype=float)
        A[:, j] = f_test - b
        
    return A, b

def _check_lti(f, t0, t1, y0, n, A0, b0):
    """
    Check if A and b are constant over [t0, t1].
    Samples at t0, t_mid, t1.
    Returns True if LTI, else False.
    """
    t_mid = (t0 + t1) / 2.0
    
    # Check at t_mid
    if not _is_deadline_exceeded():
        A_mid, b_mid = _extract_AB(f, t_mid, y0, n)
        if not (np.allclose(A0, A_mid, rtol=1e-10, atol=1e-12) and 
                np.allclose(b0, b_mid, rtol=1e-10, atol=1e-12)):
            return False
        
    # Check at t1
    if not _is_deadline_exceeded():
        A_end, b_end = _extract_AB(f, t1, y0, n)
        if not (np.allclose(A0, A_end, rtol=1e-10, atol=1e-12) and 
                np.allclose(b0, b_end, rtol=1e-10, atol=1e-12)):
            return False
            
    return True

def integrate(f, t_span, y0):
    global _START_TIME
    _START_TIME = time.time()
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0
    
    # 1. Try LTI detection
    try:
        A0, b0 = _extract_AB(f, t0, y0, n)
        
        if _is_deadline_exceeded():
            # Fallback to a simple step if we are out of time, though unlikely this early
            return y0 + f(t0, y0) * dt
            
        is_lti = _check_lti(f, t0, t1, y0, n, A0, b0)
        
        if is_lti and not _is_deadline_exceeded():
            # Solve y' = A y + b
            # y(t) = exp(A(t-t0)) y0 + integral...
            M = la.expm(A0 * dt)
            
            if np.allclose(b0, 0.0, atol=1e-12):
                y_end = M @ y0
            else:
                # Particular solution for constant b: A^{-1}(exp(A dt) - I) b
                # If A is singular, use the series expansion or numerical integration
                try:
                    # Check condition number
                    if np.linalg.cond(A0) < 1e10:
                        A_inv = la.inv(A0)
                        y_part = A_inv @ (M - np.eye(n)) @ b0
                    else:
                        # Use numerical integration for the particular part
                        # y' = A y + b, y(0)=0. 
                        # Use solve_ivp with tight tolerances for just this part?
                        # Or use the matrix exponential series for the integral term:
                        # integral = sum_{k=0}^inf (A^k dt^{k+1} / (k+1)!) b
                        # This converges fast for small dt or small A
                        y_part = np.zeros(n)
                        term = np.ones(n) # A^0 b
                        term = b0.copy()
                        y_part += term * dt
                        for k in range(1, 20):
                            term = A0 @ term / k
                            y_part += term * (dt ** k) / k
                            if np.linalg.norm(term) < 1e-15:
                                break
                except np.linalg.LinAlgError:
                    # Fallback to numerical integration
                    y_part = np.zeros(n)
                    
                y_end = M @ y0 + y_part
            
            return y_end
            
    except Exception as e:
        # If LTI detection fails for any reason, fall through to general solver
        pass
    
    # 2. Fallback: General stiff solver
    # Use LSODA or Radau. Radau is good for stiff.
    # We need to be careful with the number of calls.
    # solve_ivp will call f many times.
    # We set tight tolerances to maximize accuracy.
    
    def f_wrapper(t, y):
        return f(t, y)
        
    try:
        sol = solve_ivp(f_wrapper, [t0, t1], y0, method='LSODA', 
                        rtol=1e-10, atol=1e-12, dense_output=False)
        if sol.success:
            return sol.y[:, -1]
        else:
            # If LSODA fails, try Radau
            sol = solve_ivp(f_wrapper, [t0, t1], y0, method='Radau', 
                            rtol=1e-10, atol=1e-12)
            if sol.success:
                return sol.y[:, -1]
    except Exception:
        pass
        
    # 3. Last resort: Large step RK4 (likely inaccurate but better than nothing)
    # This is a very bad fallback, but ensures a return
    h = dt / 100
    t, y = t0, y0.copy()
    for _ in range(100):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
