"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p  # 60
    
    # Helper to validate a decomposition
    def is_valid_decomposition(U, V, W, rank):
        U_mat = np.array(U)
        V_mat = np.array(V)
        W_mat = np.array(W)
        
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i*m+j, :]
                    v_val = V_mat[j*p+k, :]
                    w_val = W_mat[k*n+i, :]
                    total = np.sum(u_val * v_val * w_val)
                    if not np.isclose(total, 1.0, atol=1e-9):
                        return False
        return True

    # 1. Establish Baseline (Schoolbook)
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_sb}
    
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    def time_remaining():
        return time_limit - (time.time() - start_time)
    
    # Search for lower rank
    # Known inspirations found rank 54 (score 1.11). We aim for <= 54.
    # Try ranks from 59 down to 45.
    min_rank = 45
    
    for r in range(rank_sb - 1, min_rank - 1, -1):
        if time_remaining() < 5.0:
            break
        
        # Number of attempts per rank
        attempts = 2000 if r > 55 else 1000
        
        for attempt in range(attempts):
            if time_remaining() < 2.0:
                break
            
            # Random sparse U and V
            # Density affects the rank of M_{i,k}
            density = 0.2 + 0.1 * (attempt % 3) # 0.2, 0.3, 0.4
            
            U_rand = np.zeros((n * m, r), dtype=float)
            V_rand = np.zeros((m * p, r), dtype=float)
            
            mask_u = np.random.rand(n * m, r) < density
            mask_v = np.random.rand(m * p, r) < density
            
            U_rand[mask_u] = np.random.choice([-1, 1], size=mask_u.sum())
            V_rand[mask_v] = np.random.choice([-1, 1], size=mask_v.sum())
            
            # Check if we can find integer/half-integer W for each (i,k) block
            W_candidate = np.zeros((p * n, r), dtype=float)
            valid_block = True
            
            for i in range(n):
                for k in range(p):
                    # Construct M_{i,k} of shape (m, r)
                    # M_{j, c} = U[i*m+j, c] * V[j*p+k, c]
                    M = np.zeros((m, r), dtype=float)
                    for j in range(m):
                        M[j, :] = U_rand[i*m+j, :] * V_rand[j*p+k, :]
                    
                    # Solve M * w = 1
                    # We want w to be integer or half-integer
                    try:
                        # Use least squares to find minimum norm solution
                        w, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    except Exception:
                        valid_block = False
                        break
                    
                    # Check if w is close to integer or half-integer
                    # Try integer
                    w_int = np.round(w)
                    if np.allclose(M @ w_int, np.ones(m), atol=1e-9):
                        W_candidate[k*n+i, :] = w_int
                    else:
                        # Try half-integer
                        w_half = np.round(w * 2) / 2.0
                        if np.allclose(M @ w_half, np.ones(m), atol=1e-9):
                            W_candidate[k*n+i, :] = w_half
                        else:
                            valid_block = False
                            break
                
                if not valid_block:
                    break
            
            if valid_block:
                # Final verification
                if is_valid_decomposition(U_rand.tolist(), V_rand.tolist(), W_candidate.tolist(), r):
                    best_result = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_candidate.tolist(), "rank": r}
                    # Return immediately if we found a good rank, or continue to find lower?
                    # The task asks for rank-minimal. But finding lower is hard.
                    # Score is seed/value. Lower rank = higher score.
                    # We break out of rank loop if we find one, but maybe we can find lower?
                    # Given time constraints, we return the first valid lower rank found, 
                    # or continue searching for lower ranks if time permits.
                    # Since we iterate r downwards, the first one found is the best for this loop iteration.
                    # We can break out of the attempt loop.
                    break
        
        # If we found a valid decomposition for rank r, we can try to find lower ranks.
        # But if we found rank 54, and we have time, we should try 53, etc.
        # The loop structure already handles this by continuing to the next r if not found.
        # If found, we update best_result and continue to next r.
        
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
# EVOLVE-BLOCK-END
