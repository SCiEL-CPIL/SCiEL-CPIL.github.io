"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify_einsum(U, V, W, tol=1e-9):
    """Use einsum for verification."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=tol)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 75
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = schoolbook_rank
    
    # Try to remove columns one by one
    def is_valid(U_mat, V_mat, W_mat):
        return _verify_einsum(U_mat, V_mat, W_mat, tol=1e-9)
    
    # Greedy column removal with multiple random orders
    current_rank = schoolbook_rank
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    for attempt in range(8):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            order = list(range(current_rank))
        elif attempt == 1:
            order = list(range(current_rank - 1, -1, -1))
        else:
            order = list(range(current_rank))
            random.seed(100 + attempt)
            random.shuffle(order)
        
        U_try = U_cur.copy()
        V_try = V_cur.copy()
        W_try = W_cur.copy()
        rank_try = current_rank
        
        for idx in order:
            if time.time() > deadline:
                break
            if idx >= rank_try:
                continue
            U_new = np.delete(U_try, idx, axis=1)
            V_new = np.delete(V_try, idx, axis=1)
            W_new = np.delete(W_try, idx, axis=1)
            
            if is_valid(U_new, V_new, W_new):
                U_try = U_new
                V_try = V_new
                W_try = W_new
                rank_try -= 1
        
        if rank_try < best_rank:
            best_rank = rank_try
            best_U = U_try.copy()
            best_V = V_try.copy()
            best_W = W_try.copy()
            U_cur = U_try.copy()
            V_cur = V_try.copy()
            W_cur = W_try.copy()
            current_rank = rank_try
    
    if best_rank < schoolbook_rank:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
