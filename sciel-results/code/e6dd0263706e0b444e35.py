import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = start_time + 100.0  # 100s margin
    
    # Try to find lower rank
    # Lower bound is max(nm, mp, pn) = 25
    min_rank = max(n * m, m * p, p * n)
    
    # We search from low rank to high rank
    # If we find a valid one, we return it immediately as it's better than schoolbook
    # To ensure we find a good one, we might want to search a range
    # But any R < 75 is an improvement.
    
    for R in range(min_rank, rank_schoolbook):
        if time.time() > deadline:
            break
            
        # Number of trials per rank
        # For small R, the system is tight. For larger R, it's looser.
        trials = 50 if R < 40 else 10
        
        for _ in range(trials):
            if time.time() > deadline:
                break
                
            # Random U, V with small integers or floats
            # Using integers helps with exactness if we use rational arithmetic, 
            # but float64 is usually sufficient for verification if we check residual.
            U_rand = np.random.randint(-2, 3, (n * m, R)).astype(float)
            V_rand = np.random.randint(-2, 3, (m * p, R)).astype(float)
            
            # Construct linear system A * W_vec = b
            # b is all ones
            b = np.ones(n * m * p)
            
            # A has shape (n*m*p, p*n*R)
            # Row idx: (i, j, k)
            # Col idx: (k, i, r)
            A = np.zeros((n * m * p, p * n * R))
            
            # Precompute indices
            # Map (i, j, k) to row
            # Map (k, i, r) to col
            
            row_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # U[i*m+j, r] * V[j*p+k, r]
                        u_val = U_rand[i * m + j, :]
                        v_val = V_rand[j * p + k, :]
                        prod = u_val * v_val # shape (R,)
                        
                        # This prod contributes to column (k, i, r)
                        # Col index = (k * n + i) * R + r
                        base_col = (k * n + i) * R
                        A[row_idx, base_col:base_col+R] = prod
                        row_idx += 1
            
            # Solve A W_vec = b
            try:
                W_vec, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                # Check residual
                if len(residuals) > 0:
                    res = residuals[0]
                else:
                    res = np.linalg.norm(A @ W_vec - b)
                
                if res < 1e-8:
                    # Valid decomposition found
                    W_mat = W_vec.reshape(p * n, R)
                    # Convert to lists
                    return {
                        "U": U_rand.tolist(),
                        "V": V_rand.tolist(),
                        "W": W_mat.tolist()
                    }
            except Exception:
                continue
                
    return best_result
# EVOLVE-BLOCK-END
