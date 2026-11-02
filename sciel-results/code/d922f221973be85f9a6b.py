import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[idx_u, idx_v, idx_w] = 1 if indices match, else 0
    # idx_u = i*m + j
    # idx_v = j*p + k
    # idx_w = k*n + i
    # Match condition: i_u == i_w, j_u == j_v, k_v == k_w
    
    T_target = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_target[idx_u, idx_v, idx_w] = 1

    # Search for better rank
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Try ranks from 30 to 40
    for target_rank in range(30, 41):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random integers
            # Use small integers to increase chance of exact solution
            U = np.random.randint(-2, 3, size=(n * m, target_rank))
            V = np.random.randint(-2, 3, size=(m * p, target_rank))
            W = np.random.randint(-2, 3, size=(p * n, target_rank))
            
            # Check if this random integer solution works
            # Compute T_computed
            # T_computed[idx_u, idx_v, idx_w] = sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
            # This is a tensor contraction
            # We can do this efficiently using einsum or loops
            # For small dimensions, loops are fine
            
            # Optimization: Check only non-zero entries of T_target?
            # No, we need to check ALL entries because T_computed must be 0 where T_target is 0.
            
            # We can check the difference
            # T_computed = np.einsum('ir,jr,kr->ijk', U, V, W)
            # diff = T_computed - T_target
            # if np.all(diff == 0): return
            
            # To speed up, we can check a random subset of entries first
            # If it passes the subset, check all
            
            subset_size = 100
            idxs_u = np.random.randint(0, n * m, subset_size)
            idxs_v = np.random.randint(0, m * p, subset_size)
            idxs_w = np.random.randint(0, p * n, subset_size)
            
            # Compute values for these indices
            # val = sum_r U[i,r] V[j,r] W[k,r]
            # This is a dot product of vectors U[i,:], V[j,:], W[k,:] element-wise?
            # No, sum_r (U[i,r] * V[j,r]) * W[k,r]
            
            valid = True
            for i in idxs_u:
                for j in idxs_v:
                    for k in idxs_w:
                        # Compute sum
                        # Using numpy dot
                        val = np.sum(U[i, :] * V[j, :] * W[k, :])
                        if val != T_target[i, j, k]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Full check
            # We can use einsum for the full check
            T_computed = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.array_equal(T_computed, T_target):
                # Found a solution
                best_rank = target_rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                # We can break early if we find a very good one
                # But we want the minimal rank, so we should continue to lower ranks?
                # The loop goes from 30 to 40, so we start with lower ranks.
                # If we find a rank 30 solution, we should try 29, 28...
                # So we should break the rank loop and return?
                # No, we should check if we can go lower.
                # But the loop is ascending.
                # So if we find rank 30, we don't know if 29 exists.
                # We should probably search descending?
                # Or search 30, then 31...
                # If we find 30, we can return it immediately?
                # The score is 45/rank. Lower rank is better.
                # So if we find 30, we should try to find 29, 28...
                # So we should iterate rank from 20 to 40.
                # If we find a solution, we record it and continue to lower ranks?
                # No, if we find rank R, we should try R-1, R-2...
                # So we should iterate from low to high, and if we find one, we can stop?
                # No, we want the MINIMAL rank.
                # So we should try 20, 21, 22...
                # If we find 25, we should try 24, 23...
                # So we should iterate from low to high, and if we find a solution, we can break?
                # Yes, because we are looking for the smallest rank.
                
                # Convert to lists
                return {"U": best_U, "V": best_V, "W": best_W}
            
    return {"U": best_U, "V": best_V, "W": best_W}
