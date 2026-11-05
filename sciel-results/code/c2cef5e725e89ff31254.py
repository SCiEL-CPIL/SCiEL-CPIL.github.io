import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.size
    dt = t1 - t0
    start = time.time()
    deadline = start + 295.0  # 35s safety margin before 330s kill

    if dt == 0:
        return y0.copy()

    # Helper to extract A(t) assuming f(t, y) = A(t) y
    # Uses n calls to f per evaluation of A(t)
    def get_A(t):
        A = np.empty((n, n), dtype=float)
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            A[:, j] = f(t, e)
        return A

    # 1. Check for time-independent linear system
    A0 = get_A(t0)
    
    # Check homogeneity: f(t, 0) should be 0
    z0 = f(t0, np.zeros(n))
    hom_err = np.linalg.norm(z0)
    
    # Check time-independence: A(t1) and A(mid) should match A0
    A1 = get_A(t1)
    Amid = get_A(0.5 * (t0 + t1))
    
    time_indep_err = max(np.linalg.norm(A0 - A1), np.linalg.norm(A0 - Amid))
    
    # If it looks like a constant linear system
    if hom_err < 1e-12 and time_indep_err < 1e-12:
        M = expm(A0 * dt)
        return M @ y0

    # 2. Check for time-dependent linear system
    # Verify linearity at a few points
    np.random.seed(42)
    y_test1 = np.random.randn(n)
    y_test2 = np.random.randn(n)
    alpha = 2.5
    
    # Test at t_mid
    t_test = 0.5 * (t0 + t1)
    A_test = get_A(t_test)
    
    pred_sum = A_test @ (y_test1 + y_test2)
    actual_sum = f(t_test, y_test1 + y_test2)
    err_sum = np.linalg.norm(pred_sum - actual_sum) / (1e-30 + np.linalg.norm(actual_sum))
    
    pred_alpha = A_test @ (alpha * y_test1)
    actual_alpha = f(t_test, alpha * y_test1)
    err_alpha = np.linalg.norm(pred_alpha - actual_alpha) / (1e-30 + np.linalg.norm(actual_alpha))
    
    # Also check homogeneity at t_test
    z_test = f(t_test, np.zeros(n))
    hom_err_test = np.linalg.norm(z_test)
    
    is_linear = (err_sum < 1e-10 and err_alpha < 1e-10 and hom_err_test < 1e-12)
    
    if is_linear:
        # Use Piecewise Constant Approximation (PCA) with Matrix Exponentials
        # This is highly accurate for linear systems.
        # Choose number of steps K.
        # We need to balance accuracy vs number of f calls (n*K) and time.
        
        # Let's estimate cost.
        # expm for 15x15 is fast (~ms).
        # f calls are the bottleneck for the score (log penalty).
        # But accuracy is the primary score component.
        
        # Strategy: Adaptive K? Or fixed large K?
        # Since we have a time deadline, let's try to maximize K within time.
        # But we also need to ensure we don't run out of time.
        
        # Let's start with a moderate K and increase if time permits.
        # Or just pick a K that is safe.
        # K=5000 means 75,000 f calls. 
        # 5000 expm calls.
        # Should be well within 295s.
        
        K = 5000
        h = dt / K
        y = y0.copy()
        
        for k in range(K):
            if time.time() > deadline:
                # Fallback: return current y
                return y
                
            t_mid = t0 + (k + 0.5) * h
            A_k = get_A(t_mid)
            M_k = expm(A_k * h)
            y = M_k @ y
            
        return y

    # 3. Fallback: Non-linear or detection failed
    # Use solve_ivp with Radau
    from scipy.integrate import solve_ivp
    
    # Try to provide Jacobian if possible? 
    # If it's not linear, we can't easily provide it.
    
    sol = solve_ivp(f, (t0, t1), y0, method='Radau', rtol=1e-10, atol=1e-12,
                    max_step=0.01)
    
    if not sol.success:
        sol = solve_ivp(f, (t0, t1), y0, method='BDF', rtol=1e-10, atol=1e-12,
                        max_step=0.01)
        
    return sol.y[:, -1]
