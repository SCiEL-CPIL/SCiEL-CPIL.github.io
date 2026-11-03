import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _verify(U, V, W, n, m, p, tol=1e-10):
    rank = U.shape[1]
    # Use einsum for efficient contraction
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Construct the target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
                
    return np.allclose(T_decomp, T_target, atol=tol)

def _trivial_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _vectorized_als(n, m, p, target_rank, max_iters, deadline, rng, T):
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    U = rng.randn(n_m, target_rank) * 0.1
    V = rng.randn(m_p, target_rank) * 0.1
    W = rng.randn(p_n, target_rank) * 0.1
    
    b_U = np.zeros((n_m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                b_U[i*m+j, k] = T[i*m+j, j*p+k, k*n+i]
                
    b_V = np.zeros((m_p, n))
    for j in range(m):
        for k in range(p):
            for i in range(n):
                b_V[j*p+k, i] = T[i*m+j, j*p+k, k*n+i]
                
    b_W = np.zeros((p_n, m))
    for k in range(p):
        for i in range(n):
            for j in range(m):
                b_W[k*n+i, j] = T[i*m+j, j*p+k, k*n+i]

    prev_residual = np.inf
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            return None
        
        # --- Update U ---
        V_reshaped = V.reshape(m, p, target_rank)
        W_reshaped = W.reshape(p, n, target_rank)
        
        V_exp = V_reshaped[:, :, np.newaxis, :]
        W_exp = W_reshaped[np.newaxis, :, :, :]
        
        prod = V_exp * W_exp
        prod = prod.transpose(2, 0, 1, 3)
        A_U = prod.reshape(n_m, p, target_rank)
        
        try:
            U, residuals, rank_u, sv = np.linalg.lstsq(A_U, b_U, rcond=None)
        except:
            pass

        # --- Update V ---
        U_reshaped = U.reshape(n, m, target_rank)
        W_reshaped = W.reshape(p, n, target_rank)
        
        U_b = U_reshaped[:, :, np.newaxis, :]
        W_b = W_reshaped.T[:, np.newaxis, :, :]
        
        prod = U_b * W_b
        prod = prod.transpose(1, 2, 0, 3)
        A_V = prod.reshape(m_p, n, target_rank)
        
        try:
            V, residuals, rank_v, sv = np.linalg.lstsq(A_V, b_V, rcond=None)
        except:
            pass

        # --- Update W ---
        U_reshaped = U.reshape(n, m, target_rank)
        V_reshaped = V.reshape(m, p, target_rank)
        
        U_b = U_reshaped[:, :, np.newaxis, :]
        V_b = V_reshaped[np.newaxis, :, :, :]
        
        prod = U_b * V_b
        prod = prod.transpose(2, 0, 1, 3)
        A_W = prod.reshape(p_n, m, target_rank)
        
        try:
            W, residuals, rank_w, sv = np.linalg.lstsq(A_W, b_W, rcond=None)
        except:
            pass
        
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        residual = np.max(np.abs(T - T_decomp))
        
        if residual < 1e-12:
            return U, V, W
            
        if residual > prev_residual * 1.1:
            break
        prev_residual = residual
        
    return None

def _find_lower_rank(n, m, p, target_rank, deadline, seed=42):
    rng = np.random.RandomState(seed)
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    T = np.zeros((n_m, m_p, p_n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    n_attempts = 20
    max_iters = 100
    
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
            
        result = _vectorized_als(n, m, p, target_rank, max_iters, deadline, rng, T)
        if result is not None:
            return result
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    best_U, best_V, best_W = _trivial_decomposition(n, m, p)
    best_rank = n * m * p
    
    candidate_ranks = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 50, 60]
    
    for target_rank in candidate_ranks:
        if target_rank >= best_rank:
            continue
        if time.time() > deadline:
            break
            
        result = _find_lower_rank(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                if best_rank == 21:
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
