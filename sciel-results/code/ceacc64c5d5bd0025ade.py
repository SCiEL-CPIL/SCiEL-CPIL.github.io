import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 110  # Safety margin
    
    best_rank = n * m * p
    best_decomp = None
    
    # Pre-compute schoolbook as fallback
    U_school = np.zeros((n * m, best_rank))
    V_school = np.zeros((m * p, best_rank))
    W_school = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_decomp = (U_school, V_school, W_school)
    
    # Try to find lower rank
    # Search from best_rank - 1 down to 40
    ranks_to_try = list(range(best_rank - 1, 39, -1))
    
    for r in ranks_to_try:
        if time.time() - start_time > timeout:
            break
            
        max_restarts = 10
        found = False
        
        for restart in range(max_restarts):
            if time.time() - start_time > timeout:
                break
                
            # Initialize
            U = np.random.randn(n, m, r) * 0.5
            V = np.random.randn(m, p, r) * 0.5
            W = np.random.randn(p, n, r) * 0.5
            
            # ALS iterations
            converged = False
            for it in range(100):
                if time.time() - start_time > timeout:
                    break
                    
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = V[j] * W[:, i, :]
                        u, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i, j, :] = u
                        
                # Update V
                for j in range(m):
                    for k in range(p):
                        B = U[:, j, :] * W[k, :, :]
                        v, residuals, rank_B, s = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j, k, :] = v
                        
                # Update W
                for k in range(p):
                    for i in range(n):
                        C = U[i, :, :] * V[:, k, :]
                        w, residuals, rank_C, s = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[k, i, :] = w
                        
                # Check error
                T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
                err = np.sum((T - 1.0) ** 2)
                
                if err < 1e-6:
                    converged = True
                    break
                    
            if converged:
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_rounded = np.einsum('ijc,jkc,kic->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(T_rounded, 1.0, atol=1e-9):
                    U_flat = U_rounded.reshape(n * m, r)
                    V_flat = V_rounded.reshape(m * p, r)
                    W_flat = W_rounded.reshape(p * n, r)
                    best_rank = r
                    best_decomp = (U_flat, V_flat, W_flat)
                    found = True
                    break
            
        if found:
            pass

    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
