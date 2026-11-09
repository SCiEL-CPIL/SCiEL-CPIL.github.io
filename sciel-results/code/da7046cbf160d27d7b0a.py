import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def _verify_fast(U, V, W, n, m, p, rank):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    Uses NumPy for speed.
    T[i,j,k] should equal 1 for all (i,j,k).
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reshape to 3D tensors:
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note the indices:
    # U index: i, m+j -> U[i,j,r]
    # V index: j, p+k -> V[j,k,r]
    # W index: k, n+i -> W[k,i,r]
    
    # We can use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
    
    # Check if all elements are 1
    return np.allclose(T, 1.0, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
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
    # The search space is vast. Random search is unlikely to find minimal rank.
    # However, with vectorized verification, we can test many candidates.
    
    # Entry set: Try small integers.
    # Note: The problem allows half-integers, so we can search integers and divide by 2 if needed,
    # but usually integer solutions exist for these small cases if we scale appropriately.
    # Let's stick to integers {-2, -1, 0, 1, 2} to allow some flexibility.
    entries = [-2, -1, 0, 1, 2]
    
    # We will iterate through target ranks.
    # It is highly probable that no rank < 45 is found by random search quickly.
    # But we must try.
    
    # Optimization: Pre-compute the index mappings for vectorized check?
    # No, einsum is fast enough.
    
    # Let's try a few random restarts for rank 44, 43, etc.
    # If we find one, we update best.
    
    # To make the search more effective, we can try to "mutate" the schoolbook solution.
    # But random is safer for correctness.
    
    # We will allocate time based on rank. Lower ranks are harder, so maybe fewer tries?
    # Or just uniform tries until deadline.
    
    # Let's try ranks from 44 down to 30.
    # We will do a fixed number of iterations per rank, determined by time.
    
    current_rank_to_try = rank_seed - 1
    
    while current_rank_to_try > 20 and time.time() < deadline:
        target_rank = current_rank_to_try
        
        # Determine number of trials based on remaining time
        # Rough estimate: einsum on (3,3,R) is very fast.
        # Let's try 5000 iterations per rank if time permits.
        
        max_iters = 5000
        found = False
        
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            # Generate random entries
            # Using list comprehension is slow. Use numpy random choice.
            U = np.random.choice(entries, size=(n*m, target_rank)).tolist()
            V = np.random.choice(entries, size=(m*p, target_rank)).tolist()
            W = np.random.choice(entries, size=(p*n, target_rank)).tolist()
            
            if _verify_fast(U, V, W, n, m, p, target_rank):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
        
        if found:
            # Found a decomposition at this rank, try to go lower
            current_rank_to_try -= 1
        else:
            # If we can't find at this rank, it might be impossible or just hard.
            # Move to next lower rank? Or give up?
            # Usually, if rank R is hard, R-1 is harder.
            # But maybe we got unlucky.
            # Let's decrement and try, but with fewer iterations if we've spent a lot of time.
            current_rank_to_try -= 1
            
            # If we've tried several ranks without success, we might want to stop
            # to avoid wasting time on impossible ranks.
            # But for now, let's just proceed. The deadline guard will stop us.
            
            # Heuristic: If we fail at rank 44, 43, 42, it's likely we won't find 41 easily either.
            # We can break if we've failed 3 consecutive ranks.
            # But let's keep it simple for now.
            
            # To prevent infinite loop on impossible ranks, we can reduce max_iters if we've spent > 50s
            if time.time() - start_time > 50.0:
                 max_iters = 500
                 if time.time() - start_time > 80.0:
                     max_iters = 100

    return best
