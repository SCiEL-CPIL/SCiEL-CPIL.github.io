import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    time_limit = 150.0
    
    U_dim = n * m
    V_dim = m * p
    W_dim = n * p
    
    # Target Tensor T (0 or 1)
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = i * p + k
                T[u, v, w] = 1
                
    target_T = 8 * T # Scaled target
    
    # Schoolbook
    U_sb = np.zeros((U_dim, rank_schoolbook), dtype=np.int32)
    V_sb = np.zeros((V_dim, rank_schoolbook), dtype=np.int32)
    W_sb = np.zeros((W_dim, rank_schoolbook), dtype=np.int32)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = i * p + k
                U_sb[u, col] = 1
                V_sb[v, col] = 1
                W_sb[w, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    def get_mismatch(U, V, W):
        # U, V, W are int arrays
        computed_T = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.sum(np.abs(computed_T - target_T))

    # Search for lower rank
    # We prioritize finding any rank < 36
    for R in range(rank_schoolbook - 1, 20, -1):
        if time.time() - start_time > time_limit - 10:
            break
        
        found = False
        num_restarts = 200
        for _ in range(num_restarts):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Initialize
            U = np.random.randint(-3, 4, (U_dim, R))
            V = np.random.randint(-3, 4, (V_dim, R))
            W = np.random.randint(-3, 4, (W_dim, R))
            
            cur_U, cur_V, cur_W = U.copy(), V.copy(), W.copy()
            cur_mis = get_mismatch(cur_U, cur_V, cur_W)
            
            if cur_mis == 0:
                found = True
                break
                
            # Local Search
            steps = 100
            for _ in range(steps):
                if time.time() - start_time > time_limit - 5:
                    break
                    
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    row = np.random.randint(U_dim)
                elif mat_idx == 1:
                    row = np.random.randint(V_dim)
                else:
                    row = np.random.randint(W_dim)
                    
                col = np.random.randint(R)
                
                # Perturbation: +/- 1 or 2
                delta = np.random.choice([-2, -1, 1, 2])
                
                old_val = 0
                if mat_idx == 0:
                    old_val = cur_U[row, col]
                    cur_U[row, col] = old_val + delta
                    new_mis = get_mismatch(cur_U, cur_V, cur_W)
                    if new_mis < cur_mis:
                        cur_mis = new_mis
                        if cur_mis == 0:
                            found = True
                            break
                    else:
                        cur_U[row, col] = old_val
                elif mat_idx == 1:
                    old_val = cur_V[row, col]
                    cur_V[row, col] = old_val + delta
                    new_mis = get_mismatch(cur_U, cur_V, cur_W)
                    if new_mis < cur_mis:
                        cur_mis = new_mis
                        if cur_mis == 0:
                            found = True
                            break
                    else:
                        cur_V[row, col] = old_val
                else:
                    old_val = cur_W[row, col]
                    cur_W[row, col] = old_val + delta
                    new_mis = get_mismatch(cur_U, cur_V, cur_W)
                    if new_mis < cur_mis:
                        cur_mis = new_mis
                        if cur_mis == 0:
                            found = True
                            break
                    else:
                        cur_W[row, col] = old_val
                
                if found:
                    break
            
            if found:
                break

        if found:
            best_rank = R
            best_U, best_V, best_W = cur_U.copy(), cur_V.copy(), cur_W.copy()
            # If we find a low rank, we might want to try to go lower?
            # The loop continues.

    # Convert to output format
    U_out = [[x / 2.0 for x in row] for row in best_U]
    V_out = [[x / 2.0 for x in row] for row in best_V]
    W_out = [[x / 2.0 for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
