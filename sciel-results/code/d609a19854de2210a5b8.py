# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    For the fixed 15D stiff linear system, infer A via basis probes,
    then use matrix exponential for high accuracy with minimal calls.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    deadline = time.time() + 280.0
    
    # Infer A by probing with standard basis vectors (n calls)
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
    
    # Check homogeneity: f(t0, 0) should be 0 (1 call)
    f_zero = f(t0, np.zeros(n))
    is_homogeneous = np.allclose(f_zero, 0.0, atol=1e-12)
    
    # Check time-independence and linearity with one combined probe (1 call)
    is_time_independent = True
    is_linear = True
    if is_homogeneous:
        rng = np.random.default_rng(42)
        y_test = rng.standard_normal(n) * 0.5
        f_test_t0 = f(t0, y_test)
        f_test_t1 = f(t1, y_test)
        
        # Time-independence: f(t1, y_test) should equal f(t0, y_test)
        if not np.allclose(f_test_t1, f_test_t0, rtol=1e-10, atol=1e-12):
            is_time_independent = False
        
        # Linearity: f(t0, y_test) should equal A @ y_test
        f_pred = A @ y_test
        if not np.allclose(f_test_t0, f_pred, rtol=1e-8, atol=1e-10):
            is_linear = False
    
    if is_linear and is_homogeneous and is_time_independent:
        T = t1 - t0
        M = T * A
        expM = expm(M)
        Y_final = expM @ y0
        
        # Quick verification with one more call if time permits
        if time.time() < deadline - 5.0:
            f_check = f(t1, Y_final)
            A_check = A @ Y_final
            if np.allclose(f_check, A_check, rtol=1e-6, atol=1e-8):
                return Y_final
        else:
            return Y_final
    
    # Fallback: use scipy's solve_ivp with a stiff method
    from scipy.integrate import solve_ivp
    
    def wrapper(t, y):
        return f(t, y)
    
    # Try BDF first for stiff systems
    sol = solve_ivp(wrapper, [t0, t1], y0, method='BDF', rtol=1e-13, atol=1e-15)
    if sol.success:
        return sol.y[:, -1]
    
    # Try Radau
    sol = solve_ivp(wrapper, [t0, t1], y0, method='Radau', rtol=1e-13, atol=1e-15)
    if sol.success:
        return sol.y[:, -1]
    
    # Try DOP853 with tight tolerances
    sol = solve_ivp(wrapper, [t0, t1], y0, method='DOP853', rtol=1e-14, atol=1e-16)
    if sol.success:
        return sol.y[:, -1]
    
    return y0
# EVOLVE-BLOCK-END
