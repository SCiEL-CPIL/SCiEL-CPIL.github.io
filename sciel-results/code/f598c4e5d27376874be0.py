import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0

    # Schoolbook solution
    r_sb = n * m * p
    U_sb = np.zeros((n * m, r_sb))
    V_sb = np.zeros((m * p, r_sb))
    W_sb = np.zeros((p * n, r_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = r_sb

    def verify_numpy(U, V, W, R):
        try:
            U_t = U.reshape(n, m, R)
            V_t = V.reshape(m, p, R)
            W_t = W.reshape(p, n, R)
            T = np.einsum('ijr,jkr,ki->ijk', U_t, V_t, W_t)
            return np.allclose(T, 1.0, atol=1e-9)
        except Exception:
            return False

    # Search for lower rank
    # Allowed values: half-integers
    vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Try ranks from 47 down to 40
    for target_rank in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Limit attempts per rank to avoid spending too much time
        # 47 is the most likely candidate
        max_attempts = 500 if target_rank == 47 else 100
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            U = np.random.choice(vals, size=(n * m, target_rank))
            V = np.random.choice(vals, size=(m * p, target_rank))
            W = np.random.choice(vals, size=(p * n, target_rank))
            
            if verify_numpy(U, V, W, target_rank):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                found = True
                break
        
        if found:
            # If we found a solution for this rank, we can stop or continue to lower ranks
            # Continuing to lower ranks is risky if we run out of time
            # But if we found 47, we can try 46 with less time
            pass
        else:
            # If we didn't find a solution for 47, it's unlikely we'll find 46
            # But we should still try if time permits
            pass

    return best_result
