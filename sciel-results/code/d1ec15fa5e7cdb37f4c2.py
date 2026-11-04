"""Matrix multiplication tensor decomposition for (2,3,10) using block decomposition."""
import time
import numpy as np
import random
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 10]

def _verify_decomposition(U, V, W, n, m, p, max_checks=5):
    """Verify if U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    random.seed(42)
    for _ in range(max_checks):
        A = np.random.uniform(-2, 2, (n, m))
        B = np.random.uniform(-2, 2, (m, p))
        C = A @ B
        
        U_reshaped = U.reshape(n, m, rank)
        u_a = np.einsum('ijr,ij->ir', U_reshaped, A)
        
        V_reshaped = V.reshape(m, p, rank)
        v_b = np.einsum('jkr,jk->kr', V_reshaped, B)
        
        W_reshaped = W.reshape(p, n, rank)
        C_computed = np.einsum('kir,ir,kr->ik', W_reshaped, u_a, v_b)
        
        if np.max(np.abs(C - C_computed)) > 1e-9:
            return False
    return True

def _find_rank_decomposition(n, m, p, target_rank, time_limit=10.0):
    """Try to find a decomposition of rank target_rank for M_{n,m,p}."""
    start_time = time.time()
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(restart)
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        params_init = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def residuals(params):
            U_flat = params[:n*m*target_rank]
            V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank]
            W_flat = params[n*m*target_rank + m*p*target_rank:]
            
            U_mat = U_flat.reshape(n*m, target_rank)
            V_mat = V_flat.reshape(m*p, target_rank)
            W_mat = W_flat.reshape(p*n, target_rank)
            
            # Vectorized residual calculation
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # We want T[i,j,k] = 1 for all i,j,k.
            
            # Reshape for einsum
            U_r = U_mat.reshape(n, m, target_rank)
            V_r = V_mat.reshape(m, p, target_rank)
            W_r = W_mat.reshape(p, n, target_rank)
            
            # Compute tensor T
            # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
            T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            
            # Residual is T - 1
            res = T.flatten() - 1.0
            return res
        
        try:
            sol = least_squares(residuals, params_init, method='lm', max_nfev=100)
            if sol.cost < 1e-12:
                U_sol = sol.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = sol.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = sol.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                # Round to nearest half-integer
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                    return U_rounded, V_rounded, W_rounded
        except Exception:
            pass
    return None

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # First, try to find a rank-12 decomposition for M_{2,3,3}
    # If found, we can build a rank-42 solution for M_{2,3,10}
    # M_{2,3,10} can be split into 3 blocks of M_{2,3,3} and 1 block of M_{2,3,1}
    # Rank(M_{2,3,1}) = 6
    
    rank_233_decomp = None
    if time.time() < DEADLINE - 30:
        rank_233_decomp = _find_rank_decomposition(2, 3, 3, 12, time_limit=20.0)
    
    if rank_233_decomp is not None:
        U_233, V_233, W_233 = rank_233_decomp
        # Build rank-42 solution
        rank = 3 * 12 + 6
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        # 3 blocks of 2x3x3
        for block in range(3):
            k_start = block * 3
            k_end = k_start + 3
            
            # Copy U_233
            U[:, col:col+12] = U_233
            
            # Copy V_233, but shift columns in V
            # V_233 has shape (3*3, 12) = (9, 12)
            # We need to place it in V which has shape (3*10, rank) = (30, rank)
            # The rows of V correspond to j*p + k.
            # For this block, k ranges from k_start to k_end.
            # So we need to map V_233[j, k_local] to V[j*10 + k_start + k_local, col + r]
            
            for j in range(3):
                for k_local in range(3):
                    k_global = k_start + k_local
                    V[j*10 + k_global, col:col+12] = V_233[j*3 + k_local, :]
            
            # Copy W_233
            # W_233 has shape (3*2, 12) = (6, 12)
            # W rows correspond to k*n + i.
            # For this block, k ranges from k_start to k_end.
            for k_local in range(3):
                k_global = k_start + k_local
                for i in range(2):
                    W[k_global*2 + i, col:col+12] = W_233[k_local*2 + i, :]
            
            col += 12
        
        # 1 block of 2x3x1
        # Rank 6 schoolbook
        for i in range(2):
            for j in range(3):
                U[i*3 + j, col] = 1
                V[j*10 + 9, col] = 1  # k=9 is the last column
                W[9*2 + i, col] = 1
                col += 1
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank-55 solution
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for pair in range(5):
        k0 = pair * 2
        k1 = pair * 2 + 1
        
        U[0, col] = 1; U[4, col] = 1
        V[k0, col] = 1; V[10+k1, col] = 1
        W[2*k0, col] = 1; W[2*k1+1, col] = 1
        col += 1
        
        U[3, col] = 1; U[4, col] = 1
        V[k0, col] = 1
        W[2*k0+1, col] = 1; W[2*k1+1, col] = -1
        col += 1
        
        U[0, col] = 1
        V[k1, col] = 1; V[10+k1, col] = -1
        W[2*k1, col] = 1; W[2*k1+1, col] = 1
        col += 1
        
        U[4, col] = 1
        V[10+k0, col] = 1; V[k0, col] = -1
        W[2*k0, col] = 1; W[2*k0+1, col] = 1
        col += 1
        
        U[0, col] = 1; U[1, col] = 1
        V[10+k1, col] = 1
        W[2*k0, col] = -1; W[2*k1, col] = 1
        col += 1
        
        U[4, col] = 1; U[0, col] = -1
        V[k0, col] = 1; V[k1, col] = 1
        W[2*k1+1, col] = 1
        col += 1
        
        U[1, col] = 1; U[4, col] = -1
        V[10+k0, col] = 1; V[10+k1, col] = 1
        W[2*k0, col] = 1; W[2*k1+1, col] = -1
        col += 1
    
    for i in range(2):
        for k in range(10):
            U[i*3+2, col] = 1
            V[20+k, col] = 1
            W[k*2+i, col] = 1
            col += 1
    
    assert col == rank, f"Column count {col} != expected rank {rank}"
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
