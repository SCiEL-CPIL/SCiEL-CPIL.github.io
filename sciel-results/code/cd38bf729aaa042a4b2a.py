import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    N = n * m  # 9
    M = m * p  # 12
    P = p * n  # 12
    
    # Build target tensor T[a, b, c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Pre-compute matricizations
    T_c = T.reshape(N * M, P)          # (108, 12)
    T_a = T.reshape(N, M * P)          # (9, 144)
    T_b = T.transpose(1, 0, 2).reshape(M, N * P)  # (12, 108)
    
    start_time = time.time()
    deadline = 130.0  # Safety margin
    
    def check_half_integer(U, V, W, tol=1e-9):
        """Check if rounding to half-integers gives exact decomposition."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify exactness
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_calc, T, atol=tol):
            return True, U_r, V_r, W_r
        return False, None, None
    
    def als_attempt(U, V, W, max_iters=200, tol=1e-10):
        """Run ALS to convergence."""
        for it in range(max_iters):
            if time.time() > deadline:
                return U, V, W, False
            
            # Update W
            E_W = np.einsum('ar,br->ab r', U, V).reshape(N * M, -1)
            try:
                W_T, _, _, _ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
                W = W_T.T
            except:
                return U, V, W, False
            
            # Update U
            E_U = np.einsum('br,cr->bc r', V, W).reshape(M * P, -1)
            try:
                U_T, _, _, _ = np.linalg.lstsq(E_U.T, T_a.T, rcond=None)
                U = U_T.T
            except:
                return U, V, W, False
            
            # Update V
            E_V = np.einsum('ar,cr->ac r', U, W).reshape(N * P, -1)
            try:
                V_T, _, _, _ = np.linalg.lstsq(E_V.T, T_b.T, rcond=None)
                V = V_T.T
            except:
                return U, V, W, False
            
            # Check convergence
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_calc - T)
            if error < tol:
                return U, V, W, True
        
        return U, V, W, False
    
    def try_polish(U, V, W, R, max_polish_iters=50):
        """Try to polish to half-integers via local optimization."""
        # First try direct rounding
        valid, U_r, V_r, W_r = check_half_integer(U, V, W)
        if valid:
            return U_r, V_r, W_r
        
        # Try iterative rounding with refinement
        for polish_it in range(max_polish_iters):
            if time.time() > deadline:
                return None
            
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Check if this is valid
            valid, U_check, V_check, W_check = check_half_integer(U_r, V_r, W_r)
            if valid:
                return U_check, V_check, W_check
            
            # If not valid, try to adjust
            # Compute residual
            T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            residual = T - T_calc
            
            # Try small perturbations to find better rounding
            best_error = np.linalg.norm(residual)
            best_U, best_V, best_W = U_r, V_r, W_r
            
            # Try flipping some entries
            for factor in [0.5, 1.0, 2.0]:
                for _ in range(3):
                    U_try = U_r + factor * np.random.randn(*U_r.shape) * 0.01
                    V_try = V_r + factor * np.random.randn(*V_r.shape) * 0.01
                    W_try = W_r + factor * np.random.randn(*W_r.shape) * 0.01
                    
                    T_try = np.einsum('ar,br,cr->abc', U_try, V_try, W_try)
                    err = np.linalg.norm(T_try - T)
                    if err < best_error:
                        best_error = err
                        best_U, best_V, best_W = U_try, V_try, W_try
            
            # Update
            U, V, W = best_U, best_V, best_W
            
            if best_error < 1e-12:
                valid, U_check, V_check, W_check = check_half_integer(U, V, W)
                if valid:
                    return U_check, V_check, W_check
        
        return None
    
    # Try ranks from low to high
    best_result = None
    
    for R in range(20, 40):
        if time.time() > deadline:
            break
        
        # Multiple random initializations
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            seed = 12345 + R * 1000 + attempt
            np.random.seed(seed)
            
            # Random initialization
            U = np.random.randn(N, R)
            V = np.random.randn(M, R)
            W = np.random.randn(P, R)
            
            # Run ALS
            U, V, W, converged = als_attempt(U, V, W)
            
            if converged:
                # Try to get half-integer solution
                polished = try_polish(U, V, W, R)
                if polished is not None:
                    U_r, V_r, W_r = polished
                    # Final verification
                    T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    if np.allclose(T_calc, T, atol=1e-9):
                        best_result = (R, U_r, V_r, W_r)
                        break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        R, U_r, V_r, W_r = best_result
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback: schoolbook decomposition (rank = n*m*p = 36)
    rank = n * m * p
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx][col] = 1
                V[b_idx][col] = 1
                W[c_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
