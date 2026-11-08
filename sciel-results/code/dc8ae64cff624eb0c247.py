"""Find rank-minimal tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def verify(U, V, W, tol=1e-8):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def clean_half_integer(arr):
        return np.round(arr * 2) / 2
    
    # Search from low rank upward
    for target_rank in range(24, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        for restart in range(30):
            if time.time() > deadline:
                break
            
            np.random.seed(12345 + restart * 7919 + target_rank * 31)
            
            U = np.random.randn(n * m, target_rank) * 0.3
            V = np.random.randn(m * p, target_rank) * 0.3
            W = np.random.randn(p * n, target_rank) * 0.3
            
            for iteration in range(300):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            U[i * m + j, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, target_rank))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            V[j * p + k, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                        except:
                            pass
                
                # Update W
                for i in range(n):
                    for k in range(p):
                        M = np.zeros((m, target_rank))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            W[k * n + i, :], _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                        except:
                            pass
                
                if verify(U, V, W, tol=1e-7):
                    U_clean = clean_half_integer(U)
                    V_clean = clean_half_integer(V)
                    W_clean = clean_half_integer(W)
                    if verify(U_clean, V_clean, W_clean, tol=1e-10):
                        best_U, best_V, best_W = U_clean, V_clean, W_clean
                        best_rank = target_rank
                        found = True
                        break
                    else:
                        # Try without cleaning
                        best_U, best_V, best_W = U, V, W
                        best_rank = target_rank
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
