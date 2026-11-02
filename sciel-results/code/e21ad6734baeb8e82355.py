"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n, m, p, max_iterations=1000, seed=42, time_limit=20):
    """
    Try to find a valid decomposition of rank r using vectorized ALS.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n*m, r)
    V = rng.randn(m*p, r)
    W = rng.randn(p*n, r)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Update U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                W_indices = np.arange(p) * n + i
                W_block = W[W_indices, :]
                A_mat = V[j*p:(j+1)*p, :] * W_block
                try:
                    u_sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                    U_new[i*m+j, :] = u_sol
                except:
                    U_new[i*m+j, :] = U[i*m+j, :]
        U = U_new
        
        # Update V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                U_indices = np.arange(n) * m + j
                U_block = U[U_indices, :]
                W_block = W[k*n:(k+1)*n, :]
                A_mat = U_block * W_block
                try:
                    v_sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                    V_new[j*p+k, :] = v_sol
                except:
                    V_new[j*p+k, :] = V[j*p+k, :]
        V = V_new
        
        # Update W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                U_block = U[i*m:(i+1)*m, :]
                V_indices = np.arange(m) * p + k
                V_block = V[V_indices, :]
                A_mat = U_block * V_block
                try:
                    w_sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                    W_new[k*n+i, :] = w_sol
                except:
                    W_new[k*n+i, :] = W[k*n+i, :]
        W = W_new
        
        # Calculate error
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        err = np.sqrt(np.mean((T - 1.0)**2))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < 1e-8:
            U_r = np.round(U * 2) / 2.0
            V_r = np.round(V * 2) / 2.0
            W_r = np.round(W * 2) / 2.0
            if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-6):
                return U_r, V_r, W_r

    if best_err < 1e-6:
        U_r = np.round(best_U * 2) / 2.0
        V_r = np.round(best_V * 2) / 2.0
        W_r = np.round(best_W * 2) / 2.0
        if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-6):
            return U_r, V_r, W_r

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110
    
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    for target_rank in range(60, 19, -1):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline:
                break
            result = _try_rank(target_rank, n, m, p, max_iterations=500, seed=seed, time_limit=10)
            if result is not None:
                U_r, V_r, W_r = result
                if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-6):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            pass

    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
