import time
import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_remaining():
        return deadline - time.time()
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify(U, V, W):
        T_decomp = np.zeros((n*m, m*p, p*n), dtype=float)
        for t in range(U.shape[1]):
            T_decomp += np.einsum('i,j,k->ijk', U[:,t], V[:,t], W[:,t])
        return np.allclose(T_decomp, T, atol=1e-8)
    
    def als_decompose(target, U_init, V_init, W_init, rank, max_iters=100, tol=1e-10):
        """Alternating Least Squares for tensor decomposition"""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iters):
            if time_remaining() < 5.0:
                break
            
            # Update U: for each t, solve min_U sum_jk (T[:,j,k] - sum_{t'≠t} U[:,t']⊗V[:,t']⊗W[:,t'])^2 * V[j,t]W[k,t]
            for t in range(rank):
                # Compute the residual excluding term t
                T_res = target.copy()
                for t2 in range(rank):
                    if t2 != t:
                        T_res -= np.einsum('i,j,k->ijk', U[:,t2], V[:,t2], W[:,t2])
                
                # The term we want: U[:,t] ⊗ V[:,t] ⊗ W[:,t]
                # We want to minimize ||T_res - U[:,t]⊗V[:,t]⊗W[:,t]||^2
                # This is equivalent to finding U[:,t] that minimizes:
                # ||vec(T_res) - U[:,t] ⊗ (V[:,t]⊗W[:,t])||^2
                # = ||M - U[:,t] ⊗ v||^2 where v = V[:,t]⊗W[:,t] and M = reshape(T_res)
                
                v = np.outer(V[:,t], W[:,t])  # shape (m*p * p*n,)
                # We want U[:,t] to approximate T_res contracted with V[:,t] and W[:,t]
                # Actually, the optimal U[:,t] given V[:,t], W[:,t] is:
                # U[:,t] = (T_res * V[:,t] * W[:,t]) / (V[:,t]^2 * W[:,t]^2) ... no
                # 
                # Let me think again. We want to minimize:
                # sum_ijk (T_res[i,j,k] - U[i,t]*V[j,t]*W[k,t])^2
                # Taking derivative wrt U[i,t]:
                # -2 * sum_jk (T_res[i,j,k] - U[i,t]*V[j,t]*W[k,t]) * V[j,t]*W[k,t] = 0
                # U[i,t] * sum_jk V[j,t]^2 * W[k,t]^2 = sum_jk T_res[i,j,k] * V[j,t] * W[k,t]
                # U[i,t] = sum_jk T_res[i,j,k] * V[j,t] * W[k,t] / (sum_j V[j,t]^2) * (sum_k W[k,t]^2)
                
                denom = np.sum(V[:,t]**2) * np.sum(W[:,t]**2)
                if denom < 1e-15:
                    continue
                U[:,t] = np.einsum('ijk,j,k->i', T_res, V[:,t], W[:,t]) / denom
            
            # Similarly update V and W
            for t in range(rank):
                T_res = target.copy()
                for t2 in range(rank):
                    if t2 != t:
                        T_res -= np.einsum('i,j,k->ijk', U[:,t2], V[:,t2], W[:,t2])
                
                denom = np.sum(U[:,t]**2) * np.sum(W[:,t]**2)
                if denom < 1e-15:
                    continue
                V[:,t] = np.einsum('ijk,i,k->j', T_res, U[:,t], W[:,t]) / denom
            
            for t in range(rank):
                T_res = target.copy()
                for t2 in range(rank):
                    if t2 != t:
                        T_res -= np.einsum('i,j,k->ijk', U[:,t2], V[:,t2], W[:,t2])
                
                denom = np.sum(U[:,t]**2) * np.sum(V[:,t]**2)
                if denom < 1e-15:
                    continue
                W[:,t] = np.einsum('ijk,i,j->k', T_res, U[:,t], V[:,t]) / denom
        
        return U, V, W
    
    # Start from rank 28 and try to reduce
    rank_fallback = 28
    U_fallback = np.zeros((n * m, rank_fallback), dtype=float)
    V_fallback = np.zeros((m * p, rank_fallback), dtype=float)
    W_fallback = np.zeros((p * n, rank_fallback), dtype=float)
    
    strassen_products = [
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}},
    ]
    
    for q in range(p // 2):
        k1 = 2 * q
        k2 = 2 * q + 1
        for r, prod in enumerate(strassen_products):
            t = q * 7 + r
            for (i, j), coeff in prod['A'].items():
                U_fallback[i * m + j, t] = coeff
            for (j, k_sub), coeff in prod['B'].items():
                k_global = k1 + k_sub
                V_fallback[j * p + k_global, t] = coeff
            for (i, k_sub), coeff in prod['C'].items():
                k_global = k1 + k_sub
                W_fallback[k_global * n + i, t] = coeff
    
    # Try to reduce rank by dropping terms and re-optimizing
    for target_rank in [27, 26, 25, 24]:
        if time_remaining() < 20.0:
            break
        
        # Initialize by taking the first target_rank terms from the rank-28 solution
        U_init = U_fallback[:, :target_rank].copy()
        V_init = V_fallback[:, :target_rank].copy()
        W_init = W_fallback[:, :target_rank].copy()
        
        # Run ALS
        U_new, V_new, W_new = als_decompose(T, U_init, V_init, W_init, target_rank, max_iters=200)
        
        if verify(U_new, V_new, W_new):
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    # Fallback
    return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}
