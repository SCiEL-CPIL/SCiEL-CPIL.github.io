import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    # T[i,j,k] should equal 1 for all (i,j,k) in the standard MM tensor
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125.0  # safety margin
    
    # Schoolbook rank
    rank_seed = n * m * p
    
    # Build schoolbook decomposition
    U_seed = [[0] * rank_seed for _ in range(n*m)]
    V_seed = [[0] * rank_seed for _ in range(m*p)]
    W_seed = [[0] * rank_seed for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i*m+j][c] = 1
                V_seed[j*p+k][c] = 1
                W_seed[k*n+i][c] = 1
                c += 1
    
    best = {"U": U_seed, "V": V_seed, "W": W_seed}
    best_rank = rank_seed
    
    # Try to find lower rank decompositions
    # Known bounds: 3x3x3 has rank at most 23, 3x3x5 is related
    # Try ranks from a reasonable range downward
    # For 3x3x5, try ranks from 44 down
    
    for target_rank in range(rank_seed - 1, 29, -1):
        if time.time() > deadline - 5.0:
            break
        
        max_iters = 5000
        found = False
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            # Generate random entries from small set
            entries = [-1, 0, 1]
            U = [[random.choice(entries) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice(entries) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice(entries) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify(U, V, W, n, m, p, target_rank):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
        
        if found:
            # Found a decomposition at this rank, try to go lower
            continue
        else:
            # If we can't find at this rank, try slightly lower
            # But for efficiency, if we fail at rank R, try R-1, etc.
            pass
    
    # Also try a local search / mutation approach from schoolbook
    # Remove one term at a time and try to fix
    if time.time() < deadline - 10.0:
        current_U = [row[:] for row in best["U"]]
        current_V = [row[:] for row in best["V"]]
        current_W = [row[:] for row in best["W"]]
        current_rank = best_rank
        
        for attempt in range(200):
            if time.time() > deadline - 5.0:
                break
            
            new_rank = current_rank - 1
            if new_rank < 20:
                break
            
            # Try to find a decomposition at new_rank by random search
            found = False
            for it in range(3000):
                if time.time() > deadline - 3.0:
                    break
                entries = [-1, 0, 1]
                U = [[random.choice(entries) for _ in range(new_rank)] for _ in range(n*m)]
                V = [[random.choice(entries) for _ in range(new_rank)] for _ in range(m*p)]
                W = [[random.choice(entries) for _ in range(new_rank)] for _ in range(p*n)]
                
                if _verify(U, V, W, n, m, p, new_rank):
                    best = {"U": U, "V": V, "W": W}
                    current_rank = new_rank
                    found = True
                    break
            
            if not found:
                break
    
    return best
