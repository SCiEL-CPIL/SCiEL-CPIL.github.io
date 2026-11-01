import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def check_decomposition(U, V, W, n, m, p):
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Check if sum_r U[i,j] V[j,k] W[k,i] == 1 for all i,j,k
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
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
                
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Try to reduce rank
    current_rank = rank
    while current_rank > 20 and time.time() < deadline:
        # Try to remove a term
        removed = False
        for r in range(current_rank):
            if time.time() > deadline:
                break
            # Try removing term r
            U_new = np.delete(U, r, axis=1)
            V_new = np.delete(V, r, axis=1)
            W_new = np.delete(W, r, axis=1)
            if check_decomposition(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                current_rank -= 1
                removed = True
                break
        if not removed:
            break
            
    best_rank = current_rank
    best_U = U
    best_V = V
    best_W = W
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
