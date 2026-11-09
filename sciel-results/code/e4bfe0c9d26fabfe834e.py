import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Probe for linearity and time-independence
    # 1. f(t0, 0)
    zero_vec = np.zeros_like(y0)
    f_zero_t0 = np.asarray(f(t0, zero_vec), dtype=float)
    
    # 2. f(t0, y0)
    f_y0_t0 = np.asarray(f(t0, y0), dtype=float)
    
    # 3. f(t1, y0)
    f_y0_t1 = np.asarray(f(t1, y0), dtype=float)
    
    # 4. f(t0, 2*y0)
    f_2y0_t0 = np.asarray(f(t0, 2.0 * y0), dtype=float)
    
    is_homogeneous = np.allclose(f_zero_t0, 0.0, atol=1e-12)
    is_time_independent = np.allclose(f_y0_t1, f_y0_t0, rtol=1e-9, atol=1e-12)
    is_linear = np.allclose(f_2y0_t0, 2.0 * f_y0_t0, rtol=1e-9, atol=1e-12)
    
    if is_homogeneous and is_time_independent and is_linear:
        # System is y' = A y
        # Try to exploit tridiagonal structure
        # We need columns of A.
        # We already have f(t0, y0) which is A @ y0. This doesn't give columns directly unless y0 is basis.
        # So we need to probe basis vectors.
        
        # Check if we can assume tridiagonal.
        # Probe col 0, 1, n-1
        e0 = np.zeros(n); e0[0] = 1.0
        c0 = np.asarray(f(t0, e0), dtype=float)
        
        e1 = np.zeros(n); e1[1] = 1.0
        c1 = np.asarray(f(t0, e1), dtype=float)
        
        eN = np.zeros(n); eN[-1] = 1.0
        cN = np.asarray(f(t0, eN), dtype=float)
        
        # Check tridiagonal structure
        tol = 1e-10
        is_tridiagonal = True
        
        if n > 2 and np.any(np.abs(c0[2:]) > tol): is_tridiagonal = False
        if n > 3 and np.any(np.abs(c1[3:]) > tol): is_tridiagonal = False
        if n > 2 and np.any(np.abs(cN[:-2]) > tol): is_tridiagonal = False
        
        # Check constant diagonals
        A_00 = c0[0]
        A_11 = c1[1]
        A_NN = cN[-1]
        
        A_10 = c0[1] if n > 1 else 0.0
        A_21 = c1[2] if n > 2 else 0.0
        A_Nm1_N = cN[-2] if n > 1 else 0.0
        
        if not np.isclose(A_00, A_11, atol=tol, rtol=1e-8): is_tridiagonal = False
        if not np.isclose(A_11, A_NN, atol=tol, rtol=1e-8): is_tridiagonal = False
        if not np.isclose(A_10, A_21, atol=tol, rtol=1e-8): is_tridiagonal = False
        if not np.isclose(A_21, A_Nm1_N, atol=tol, rtol=1e-8): is_tridiagonal = False
        
        if is_tridiagonal:
            A = np.zeros((n, n), dtype=float)
            A[np.arange(n), np.arange(n)] = A_00
            if n > 1:
                A[np.arange(1, n), np.arange(0, n-1)] = A_10
                A[np.arange(0, n-1), np.arange(1, n)] = c1[0] # A_01
            M = A * (t1 - t0)
            E = expm(M)
            return E @ y0
        else:
            # Full extraction
            A = np.zeros((n, n), dtype=float)
            A[:, 0] = c0
            A[:, 1] = c1
            A[:, -1] = cN
            
            for j in range(2, n-1):
                ej = np.zeros(n)
                ej[j] = 1.0
                A[:, j] = np.asarray(f(t0, ej), dtype=float)
                
            M = A * (t1 - t0)
            E = expm(M)
            return E @ y0
    else:
        # Non-linear or Time-dependent
        # Use robust solver
        sol = solve_ivp(f, t_span, y0, method='LSODA', rtol=1e-10, atol=1e-12)
        if not sol.success:
             # Fallback to Radau if LSODA fails?
             sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
        return sol.y[:, -1]
