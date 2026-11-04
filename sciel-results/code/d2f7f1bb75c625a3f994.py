import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]

def _build_tensor(n, m, p):
    """
    Build the multiplication tensor T of shape (n*m, m*p, p*n).
    T[i, j, k] = 1 if i, j, k correspond to a valid multiplication term contributing to output k.
    
    Indexing:
    i = r * m + c   (A row r, col c)
    j = c' * p + k' (B row c', col k')
    k = k'' * n + r'' (C row r'', col k'')
    
    For standard matrix multiplication:
    C_{r, k''} = sum_{c} A_{r, c} B_{c, k''}
    
    So, a term A_{r, c} B_{c, k''} contributes to C_{r, k''}.
    i = r*m + c
    j = c*p + k''
    k = k''*n + r
    
    T[i, j, k] = 1
    """
    T = np.zeros((n * m, m * p, p * n))
    for r in range(n):
        for c in range(m):
            for k in range(p):
                i = r * m + c
                j = c * p + k
                kk = k * n + r
                T[i, j, kk] = 1.0
    return T

def _strassen_decomposition(n, m, p):
    """
    Construct a rank-28 decomposition for 2x4x4 using heterogeneous block structure:
    - Rank-10 core for 2x2x4 sub-problem (A_left * B[:, 0:4] -> C_left)
    - Rank-9 Strassen block for A_right * B[2:4, 0:2] -> C_left
    - Rank-9 Strassen block for A_left * B[0:2, 2:4] -> C_right  
    - Rank-9 Strassen block for A_right * B[2:4, 2:4] -> C_right
    Total: 10 + 9 + 9 + 9 = 37... wait, that's too many.
    
    Let me reconsider: The strategy is rank-10 core + two rank-9 blocks = 28.
    So: rank-10 for 2x2x4, rank-9 for one 2x2x2, rank-9 for another 2x2x2.
    But we have four 2x2x2 blocks total. 
    
    Actually, re-reading: "rank-10 core representing the 2x2x4 sub-problem" 
    and "two independent rank-9 Strassen blocks for the remaining 2x2x2 sub-problems"
    
    So the 2x2x4 uses rank 10, and we have two 2x2x2 blocks using rank 9 each.
    But 2x4x4 has four 2x2x2 blocks. Unless the 2x2x4 combines two of them?
    
    Let me think differently: 
    C_left (2x2) = A_left(2x2)*B11(2x2) + A_right(2x2)*B21(2x2)
    C_right (2x2) = A_left(2x2)*B12(2x2) + A_right(2x2)*B22(2x2)
    
    The 2x2x4 sub-problem could be: A_left(2x2) * B_left(2x4) -> C_left(2x2)
    where B_left = [B11, B21] (rows 0-1 and 2-3, cols 0-1)
    
    Then the remaining are:
    A_right * B21 -> part of C_left (but B21 is already in B_left)
    A_left * B12 -> part of C_right
    A_right * B22 -> part of C_right
    
    This doesn't quite work. Let me just implement the original 4-block Strassen 
    which gives rank 28, and focus on the optimization improvements.
    """
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Standard Strassen 2x2 patterns
    patterns = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]
    
    col_idx = 0
    for a_off, b_r_off, b_c_off, c_off in blocks:
        for pat in patterns:
            for local_i, coeff in pat['a'].items():
                r = local_i // 2
                c = local_i % 2
                global_i = r * m + (a_off + c)
                U[global_i, col_idx] = coeff
            
            for local_j, coeff in pat['b'].items():
                r = local_j // 2
                c = local_j % 2
                global_j = (b_r_off + r) * p + (b_c_off + c)
                V[global_j, col_idx] = coeff
            
            for local_k, coeff in pat['c'].items():
                r = local_k // 2
                c = local_k % 2
                global_k = c_off + c
                global_i = r
                W[global_k * n + global_i, col_idx] = coeff
            
            col_idx += 1
            
    return U, V, W

def _try_optimize_rank(T, n, m, p, target_rank, seed=0, U_init=None, V_init=None, W_init=None):
    """
    Attempt to find a decomposition of rank 'target_rank' for tensor T.
    Uses multiple optimization strategies with time limits.
    Returns (U, V, W) if found with low error, else None.
    """
    start_time = time.time()
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    T_flat = T.flatten()
    
    def compute_residuals(U, V, W):
        T_calc = np.einsum('it,jt,kt->ijk', U, V, W)
        return (T_calc - T).flatten()
    
    def verify(U, V, W, atol=1e-9):
        T_check = np.einsum('it,jt,kt->ijk', U, V, W)
        return np.allclose(T_check, T, atol=atol)
    
    # Strategy 1: Random initialization
    for attempt in range(3):
        if time.time() - start_time > 15:
            break
        np.random.seed(seed + attempt)
        U = np.random.randn(dim_u, target_rank)
        V = np.random.randn(dim_v, target_rank)
        W = np.random.randn(dim_w, target_rank)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def residuals(x):
            U_l = x[:dim_u * target_rank].reshape(dim_u, target_rank)
            V_l = x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
            W_l = x[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
            return compute_residuals(U_l, V_l, W_l)
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=5000, 
                                  xtol=1e-10, ftol=1e-10, gtol=1e-10)
            
            if result.cost < 1e-12:
                U_sol = result.x[:dim_u * target_rank].reshape(dim_u, target_rank)
                V_sol = result.x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
                W_sol = result.x[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
                
                if verify(U_sol, V_sol, W_sol):
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
    
    # Strategy 2: Initialize from Strassen and project
    if U_init is not None and V_init is not None and W_init is not None:
        if time.time() - start_time > 30:
            return None
        
        # Take first target_rank columns
        U = U_init[:, :target_rank].copy()
        V = V_init[:, :target_rank].copy()
        W = W_init[:, :target_rank].copy()
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def residuals(x):
            U_l = x[:dim_u * target_rank].reshape(dim_u, target_rank)
            V_l = x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
            W_l = x[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
            return compute_residuals(U_l, V_l, W_l)
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=5000,
                                  xtol=1e-10, ftol=1e-10, gtol=1e-10)
            
            if result.cost < 1e-12:
                U_sol = result.x[:dim_u * target_rank].reshape(dim_u, target_rank)
                V_sol = result.x[dim_u * target_rank : dim_u * target_rank + dim_v * target_rank].reshape(dim_v, target_rank)
                W_sol = result.x[dim_u * target_rank + dim_v * target_rank :].reshape(dim_w, target_rank)
                
                if verify(U_sol, V_sol, W_sol):
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
    
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 120.0  # Safety margin
    
    # 1. Get the known rank-28 solution
    U28, V28, W28 = _strassen_decomposition(n, m, p)
    T = _build_tensor(n, m, p)
    
    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28
    
    # 2. Try to reduce rank - focus on most promising ranks first
    # Known lower bounds suggest rank >= 16, try 20-27 first as more likely
    target_ranks = [20, 21, 22, 23, 24, 25, 26, 27, 18, 19, 17, 16]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:  # Keep 20s safety margin
            break
            
        # Try multiple seeds
        for seed in range(3):
            if time.time() > deadline - 15:
                break
                
            res = _try_optimize_rank(T, n, m, p, target_rank, seed=seed,
                                     U_init=U28, V_init=V28, W_init=W28)
            if res is not None:
                U_sol, V_sol, W_sol = res
                
                # Try to snap to half-integers
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                T_check = np.einsum('it,jt,kt->ijk', U_snap, V_snap, W_snap)
                if np.allclose(T_check, T, atol=1e-9):
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    best_rank = target_rank
                    break
                else:
                    # Try without snapping
                    if np.allclose(np.einsum('it,jt,kt->ijk', U_sol, V_sol, W_sol), T, atol=1e-9):
                        best_U, best_V, best_W = U_sol, V_sol, W_sol
                        best_rank = target_rank
                        break
        
        if best_rank == target_rank:
            # Found solution, continue to try lower ranks if time permits
            if time.time() > deadline - 10:
                break
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

if __name__ == "__main__":
    res = construct()
    print("Rank used:", len(res['U'][0]))
