import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Time budget
    start_time = time.time()
    time_limit = 140  # Leave 10s margin
    
    # Try to find a lower rank
    # We search over r from 16 to best_rank - 1
    # For each r, we do a few random trials
    
    # To speed up, we can check validity on a subset of indices first?
    # Or just full check. 8*16*8 = 1024 entries.
    # Reconstruction cost is O(r * n * m * p) = O(r * 128).
    # For r=28, cost is ~3500 ops. Very fast.
    
    # We use random matrices with entries in {-1, 0, 1}
    # If we find a valid one, we return it.
    
    # Note: The decomposition must be exact.
    # Random search might not find it if it's rare.
    # But for small ranks, it might work.
    
    # Let's try r from 20 to 31
    for r in range(20, best_rank):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random instances for this rank
        # Number of trials depends on time left
        trials = 50
        for _ in range(trials):
            if time.time() - start_time > time_limit:
                break
            
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p * n)]
            
            # Check if sum_c U[:, :, c] * V[:, :, c] * W[:, :, c] == T
            # T[i, j, k] = 1 for all i, j, k
            
            # We can vectorize this check
            # U is (n*m, r), V is (m*p, r), W is (p*n, r)
            # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
            
            U_arr = np.array(U).reshape(n, m, r)
            V_arr = np.array(V).reshape(m, p, r)
            W_arr = np.array(W).reshape(p, n, r)
            
            # Compute sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
            # This is an einsum: 'ijc,jkc,kic->ik' ? No, we need i,j,k.
            # Result[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
            
            # Note: W index is k*n + i. So W[k, i, c].
            
            # Let's use np.einsum
            # 'ijc,jkc,kic->ijk'
            try:
                reconstructed = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
                # Check if reconstructed is all 1s
                if np.all(reconstructed == 1):
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
                    # If we found a low rank, we might want to continue to find even lower?
                    # But lower is harder. We can break if r is close to lower bound.
                    # For now, we update best and continue to lower r?
                    # The loop goes from 20 to 31. If we find 28, we should try 27, 26...
                    # So we don't break the outer loop, but we update best_rank.
                    # However, the outer loop iterates r upwards.
                    # So if we find 28, we will try 29, 30... which are worse.
                    # We should iterate r downwards from 31 to 16.
                    
            except Exception:
                pass
                
    # If we didn't find anything better, best_result is still schoolbook.
    # Wait, the loop above goes from 20 to 31. If I find 28, I should stop trying higher ranks.
    # And I should try lower ranks.
    
    # Let's restructure: Iterate r from 16 to 31.
    # If found, update best and continue to lower r?
    # Actually, if I find rank 28, I should try to find 27, 26...
    # So I should iterate r from 16 up to 31.
    # If I find a solution at r, I update best_rank to r.
    # I continue searching for smaller r.
    
    # Let's rewrite the search loop properly.
    
    # Clear best to schoolbook
    best_rank = rank_schoolbook
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Search r from 16 to 31
    for r in range(16, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
        
        # If we already found a better rank, we only care about finding even better ones.
        # So if r >= best_rank, we can skip? No, we are iterating upwards.
        # If we found 28, and we are at 29, we skip.
        if r >= best_rank:
            continue
            
        # Try random trials
        # More trials for smaller r? Or fixed?
        trials = 100
        for _ in range(trials):
            if time.time() - start_time > time_limit:
                break
                
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p * n)]
            
            U_arr = np.array(U).reshape(n, m, r)
            V_arr = np.array(V).reshape(m, p, r)
            W_arr = np.array(W).reshape(p, n, r)
            
            try:
                reconstructed = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
                if np.all(reconstructed == 1):
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
                    # Found a better one, continue to try lower r
                    break # Break inner loop, continue outer loop to lower r
            except Exception:
                pass
                
    return best_result
