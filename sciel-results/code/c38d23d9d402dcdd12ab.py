import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    def extract_ab(t):
        b_t = np.array(f(t, np.zeros(n)), dtype=float)
        A_t = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            fi = np.array(f(t, ei), dtype=float)
            A_t[:, i] = fi - b_t
        return A_t, b_t

    try:
        # 1. Check if constant
        A0, b0 = extract_ab(t0)
        A1, b1 = extract_ab(t1)
        
        if np.allclose(A0, A1, atol=1e-8, rtol=1e-8) and np.allclose(b0, b1, atol=1e-8, rtol=1e-8):
            # Constant system
            M = np.zeros((n + 1, n + 1))
            M[:n, :n] = A0
            M[:n, n] = b0
            
            Z0 = np.zeros(n + 1)
            Z0[:n] = y0
            Z0[n] = 1.0
            
            Z_end = expm(M) @ Z0
            return Z_end[:n]
            
        # 2. Time-varying: Use solve_ivp with Radau
        h_jac = 1e-7
        
        def jac(t, y):
            # Re-use logic to compute A(t)
            # We don't need b(t) for Jacobian
            y_center = y
            # We can use a reference point y_ref to compute A
            # A is independent of y for linear systems
            # But f(t, y) = A(t)y + b(t)
            # f(t, y + h*e_j) - f(t, y) = A(t) h e_j
            # So A(t)[:, j] = (f(t, y + h*e_j) - f(t, y)) / h
            f_center = np.array(f(t, y_center), dtype=float)
            A = np.zeros((n, n))
            for j in range(n):
                y_pert = y_center.copy()
                y_pert[j] += h_jac
                f_pert = np.array(f(t, y_pert), dtype=float)
                A[:, j] = (f_pert - f_center) / h_jac
            return A

        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=jac,
            rtol=1e-12, 
            atol=1e-12,
            max_step=0.05 # Ensure enough resolution
        )
        
        if sol.success:
            return sol.y[:, -1]
            
    except Exception:
        pass

    # Fallback: RK4 with moderate steps
    N = 1000
    h = (t1 - t0) / N
    t = t0
    y = y0.copy()
    for _ in range(N):
        k1 = np.array(f(t, y), dtype=float)
        k2 = np.array(f(t + h/2, y + h/2 * k1), dtype=float)
        k3 = np.array(f(t + h/2, y + h/2 * k2), dtype=float)
        k4 = np.array(f(t + h, y + h * k3), dtype=float)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y
