import time
import numpy as np
import random

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Strategy: Find rank-22 decomposition for M_{2,4,3}
    # Then combine two to get rank-44 for M_{2,4,6}
    
    # Dimensions for sub-problem
    n_sub = 2
    m_sub = 4
    p_sub = 3
    rank_sub = 22
    
    # Try to find the sub-decomposition
    sub_result = _find_sub_decomposition(n_sub, m_sub, p_sub, rank_sub, max_time=100.0)
    
    if sub_result is not None:
        U_sub, V_sub, W_sub = sub_result
        # Construct full M_{2,4,6} decomposition
        # U: (2*4, 44)
        # V: (4*6, 44)
        # W: (6*2, 44)
        
        U_full = np.zeros((n * m, 2 * rank_sub), dtype=float)
        V_full = np.zeros((m * p, 2 * rank_sub), dtype=float)
        W_full = np.zeros((p * n, 2 * rank_sub), dtype=float)
        
        # First block (k = 0, 1, 2)
        # U is same for both blocks? No, U depends on i,j. It is the same structure.
        # We can reuse the same U values for the first 22 columns and the second 22 columns?
        # Wait, the decomposition is a sum of rank-1 tensors.
        # Tensor T_ijk = sum_r U_ijr V_jkr W_kir.
        # For k in 0..2, we use terms r=0..21.
        # For k in 3..5, we use terms r=22..43.
        # U_ijr for r=0..21 is U_sub_ijr.
        # U_ijr for r=22..43 is U_sub_ij(r-22). (We can reuse the same U pattern)
        U_full[:, :rank_sub] = U_sub
        U_full[:, rank_sub:] = U_sub
        
        # V_jkr
        # For k in 0..2, V_full[j, k, r] = V_sub[j, k, r] for r=0..21.
        # For k in 3..5, V_full[j, k, r] = V_sub[j, k-3, r-22] for r=22..43.
        # Note: V_sub has shape (4, 3, 22).
        for j in range(m_sub):
            for k in range(p_sub):
                V_full[j * p + k, :rank_sub] = V_sub[j, k, :]
                V_full[j * p + (k + p_sub), rank_sub:] = V_sub[j, k, :]
        
        # W_kir
        # For k in 0..2, W_full[k, i, r] = W_sub[k, i, r] for r=0..21.
        # For k in 3..5, W_full[k, i, r] = W_sub[k-3, i, r-22] for r=22..43.
        for k in range(p_sub):
            for i in range(n_sub):
                W_full[k * n + i, :rank_sub] = W_sub[k, i, :]
                W_full[(k + p_sub) * n + i, rank_sub:] = W_sub[k, i, :]
        
        # Verify full decomposition
        if _verify_full(n, m, p, U_full, V_full, W_full):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        else:
            # Should not happen if sub is valid
            pass

    # Fallback to schoolbook
    return _get_schoolbook(n, m, p)

def _get_schoolbook(n, m, p):
    nm = n * m
    mp = m * p
    pn = p * n
    rank = n * m * p
    
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_full(n, m, p, U, V, W):
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _find_sub_decomposition(n, m, p, rank, max_time=100.0):
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    
    # We search for integer solutions U', V', W' such that sum U'V'W' = 8
    # This corresponds to U=U'/2, V=V'/2, W=W'/2
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Try a few restarts
    for restart in range(50):
        if time.time() - start_time > max_time:
            break
        
        # Initialize with small random integers
        # Scale: values around 1 or 2
        # For rank 22, we need to be careful.
        U = np.random.randint(-2, 3, (nm, rank))
        V = np.random.randint(-2, 3, (mp, rank))
        W = np.random.randint(-2, 3, (pn, rank))
        
        # Reshape for einsum
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        cooling = 0.99
        
        current_error = float('inf')
        
        for step in range(5000):
            if time.time() - start_time > max_time:
                break
                
            # Calculate current error
            X = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            error = np.sum((X - 8.0)**2)
            
            if error < 1e-9:
                if _verify_sub_exact(n, m, p, U, V, W):
                    return (U / 2.0, V / 2.0, W / 2.0)
            
            # Pick a random variable to change
            mat_idx = np.random.choice(3)
            
            if mat_idx == 0:
                idx = np.random.choice(nm)
                col = np.random.choice(rank)
                old_val = U[idx, col]
                delta = np.random.choice([-1, 1])
                new_val = old_val + delta
                U[idx, col] = new_val
                U_r[idx // m, idx % m, col] = new_val
                
            elif mat_idx == 1:
                idx = np.random.choice(mp)
                col = np.random.choice(rank)
                old_val = V[idx, col]
                delta = np.random.choice([-1, 1])
                new_val = old_val + delta
                V[idx, col] = new_val
                V_r[idx // p, idx % p, col] = new_val
                
            else:
                idx = np.random.choice(pn)
                col = np.random.choice(rank)
                old_val = W[idx, col]
                delta = np.random.choice([-1, 1])
                new_val = old_val + delta
                W[idx, col] = new_val
                W_r[idx // n, idx % n, col] = new_val
                
            # Calculate new error
            X_new = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            new_error = np.sum((X_new - 8.0)**2)
            
            # Accept if better, or with probability exp(-(new_error - error)/T)
            if new_error < error:
                error = new_error
            else:
                prob = np.exp(-(new_error - error) / T)
                if np.random.random() < prob:
                    error = new_error
                else:
                    # Revert
                    if mat_idx == 0:
                        U[idx, col] = old_val
                        U_r[idx // m, idx % m, col] = old_val
                    elif mat_idx == 1:
                        V[idx, col] = old_val
                        V_r[idx // p, idx % p, col] = old_val
                    else:
                        W[idx, col] = old_val
                        W_r[idx // n, idx % n, col] = old_val
                        
            T *= cooling
            if T < T_min:
                T = T_min
                
    return None

def _verify_sub_exact(n, m, p, U, V, W):
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
                if s != 8:
                    return False
    return True
