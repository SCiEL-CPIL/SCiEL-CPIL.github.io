import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target Tensor
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_target[alpha, beta, gamma] = 1
    
    # Schoolbook Solution (Rank 50)
    best_rank = n * m * p
    best_U = np.zeros((n * m, best_rank), dtype=np.int64)
    best_V = np.zeros((m * p, best_rank), dtype=np.int64)
    best_W = np.zeros((p * n, best_rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Search for lower rank
    # Strategy: Fix R-1 random terms, check if residual is rank 1
    for R in range(49, 24, -1):
        if time.time() > deadline:
            break
        
        # Adjust attempts based on time
        # Each attempt is very fast (~1ms?)
        # Let's try 5000 attempts per rank initially, then reduce if time is tight
        max_attempts = 5000
        found = False
        
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate R-1 random terms
            # Use small integers to keep residual small and exact
            U_prev = np.random.randint(-2, 3, (n * m, R - 1))
            V_prev = np.random.randint(-2, 3, (m * p, R - 1))
            W_prev = np.random.randint(-2, 3, (p * n, R - 1))
            
            # Compute partial sum
            # T_calc = sum_{r=0}^{R-2} U_prev[:,r] * V_prev[:,r] * W_prev[:,r]
            # Vectorized:
            T_calc = np.einsum('ar,br,cr->abc', U_prev, V_prev, W_prev)
            
            T_res = T_target - T_calc
            
            # Check if T_res is rank 1
            # Find a non-zero element
            # If all zero, then R-1 is sufficient (should have been found earlier? or R-1 is the rank)
            # But we are looking for exactly R. If T_res is 0, then R-1 works.
            # We iterate R downwards, so if we find T_res=0, we should return R-1.
            # But our loop is "fix R-1, solve 1". If T_res is 0, we don't need the 1st term.
            # So we can just return R-1.
            if not np.any(T_res):
                # Rank is R-1
                best_rank = R - 1
                best_U = U_prev
                best_V = V_prev
                best_W = W_prev
                found = True
                break
            
            # Find first non-zero index
            # Use nonzero to get first index
            nz = np.nonzero(T_res)
            if len(nz[0]) == 0:
                continue
            i0, j0, k0 = nz[0][0], nz[1][0], nz[2][0]
            
            # Extract vectors
            a = T_res[:, j0, k0]
            b = T_res[i0, :, k0]
            c = T_res[i0, j0, :]
            
            # Check if T_res == a * b * c
            # T_res[i,j,k] == a[i] * b[j] * c[k]
            # We can check this by reconstructing and comparing
            # Or check a few random points? No, must be exact.
            # Reconstructing is O(nmp) = 50. Very fast.
            T_check = np.einsum('i,j,k->ijk', a, b, c)
            
            if np.array_equal(T_res, T_check):
                # It is rank 1!
                # The full solution is [U_prev, a], [V_prev, b], [W_prev, c]
                best_rank = R
                best_U = np.hstack((U_prev, a.reshape(-1, 1)))
                best_V = np.hstack((V_prev, b.reshape(-1, 1)))
                best_W = np.hstack((W_prev, c.reshape(-1, 1)))
                found = True
                break
        
        if found:
            # Found a solution for rank R (or R-1)
            # Since we iterate R downwards, the first found is the minimal found so far.
            # We can stop searching lower ranks if we are confident, or continue to see if we can go lower?
            # The loop continues to lower R.
            # If we found R, we should try R-1 in the next iteration of the outer loop.
            # So we just `continue` to the next R.
            continue
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
