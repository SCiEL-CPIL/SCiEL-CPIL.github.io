import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    U_s = np.array(U_school, dtype=float)
    V_s = np.array(V_school, dtype=float)
    
    # Target tensor T (all ones)
    # We want sum_r U[:, :, r] * V[:, :, r] * W[:, :, r] = T
    # T[i, j, k] = 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    start_time = time.time()
    deadline = 100.0  # 100s limit to be safe
    
    # Try ranks from 46 up to 53
    # 46 is the theoretical lower bound (2 * 23)
    for target_rank in range(46, rank_school):
        if time.time() > deadline:
            break
        
        # Number of trials per rank
        trials = 10 if target_rank == 46 else 5
        
        for trial in range(trials):
            if time.time() > deadline:
                break
                
            # Select target_rank columns from schoolbook
            # For diversity, use random selection
            indices = np.random.choice(rank_school, target_rank, replace=False).tolist()
            
            U_sub = U_s[:, indices]
            V_sub = V_s[:, indices]
            
            # Solve for W
            # Variables: W[k, i, r] for k in 0..p-1, i in 0..n-1, r in 0..target_rank-1
            num_vars = p * n * target_rank
            
            A = np.zeros((n * m * p, num_vars), dtype=float)
            b = np.zeros(n * m * p, dtype=float)
            
            # Fill A and b
            # Row: (i, j, k)
            # Col: (k, i, r)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = i * m * p + j * p + k
                        b[row] = 1.0
                        for r in range(target_rank):
                            col_idx = k * n * target_rank + i * target_rank + r
                            A[row, col_idx] = U_sub[i * m + j, r] * V_sub[j * p + k, r]
            
            try:
                W_sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            # Check for integer/half-integer solution
            W_flat = W_sol.flatten()
            W_rounded = np.round(W_flat * 2) / 2.0
            
            # Verify exactness
            W_rounded_matrix = W_rounded.reshape(p, n, target_rank)
            
            # Compute error
            # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
            # Vectorized check
            # U_sub is (n*m, target_rank)
            # V_sub is (m*p, target_rank)
            # W_rounded_matrix is (p, n, target_rank)
            
            # Reshape U to (n, m, target_rank)
            U_reshaped = U_sub.reshape(n, m, target_rank)
            # Reshape V to (m, p, target_rank)
            V_reshaped = V_sub.reshape(m, p, target_rank)
            
            # Compute sum over r
            # Term[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
            # Note: W index is (k, i, r).
            
            # We can do this efficiently with einsum or loops
            # Loop over k, i to be safe with indices
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_rounded_matrix[k, i, r]
                        if abs(val - 1.0) > 1e-9:
                            error = 1.0
                            break
                    if error > 0:
                        break
                if error > 0:
                    break
            
            if error < 1e-9:
                U_new = U_sub.tolist()
                V_new = V_sub.tolist()
                W_new = W_rounded_matrix.tolist()
                
                return {"U": U_new, "V": V_new, "W": W_new}
            
    return best_result
# EVOLVE-BLOCK-END
