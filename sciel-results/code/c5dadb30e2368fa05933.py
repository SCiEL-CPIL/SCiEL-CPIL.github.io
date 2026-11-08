def _try_find_rank_10_block():
    import random
    import numpy as np
    
    n, m, p = 2, 3, 2
    rank = 10
    
    # Indices for U, V, W
    # U: i*m + j (6)
    # V: j*p + k (6)
    # W: k*n + i (4)
    
    # Equations: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # for i in 0..1, j in 0..2, k in 0..1
    
    for _ in range(1000): # Limited attempts
        U = np.zeros((6, rank), dtype=int)
        V = np.zeros((6, rank), dtype=int)
        W = np.zeros((4, rank), dtype=int)
        
        # Random integers
        for r in range(rank):
            U[:, r] = random.randint(-2, 2, 6)
            V[:, r] = random.randint(-2, 2, 6)
            W[:, r] = random.randint(-2, 2, 4)
            
        # Check
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :] * V[v_idx, :], W[w_idx, :])
                    if val != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U, V, W
            
    return None
