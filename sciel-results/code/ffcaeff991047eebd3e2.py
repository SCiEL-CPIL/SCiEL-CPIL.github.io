# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    
    def schoolbook():
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = rank_schoolbook
    
    def check_decomposition_np(U, V, W):
        """Check if U, V, W form a valid decomposition using NumPy."""
        rank = U.shape[1]
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Reshape U to (n, m, rank)
        U_reshaped = U.reshape(n, m, rank)
        # Reshape V to (m, p, rank)
        V_reshaped = V.reshape(m, p, rank)
        # Reshape W to (p, n, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # Compute the tensor contraction
        # Result shape should be (n, m, p) and all values should be 1
        # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        
        # We can do this efficiently:
        # Step 1: Multiply U and V element-wise along the last axis? No.
        # We need to sum over c.
        
        # Let's use einsum for clarity and speed
        # U_reshaped[i, j, c], V_reshaped[j, k, c], W_reshaped[k, i, c]
        # Result[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        
        # np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        result = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        
        # Check if all values are 1
        if not np.allclose(result, 1.0, atol=1e-9):
            return False
        return True

    # Strategy: Try to find a decomposition with rank < 80 using a structured approach.
    # We will try random sparse matrices with {-1, 0, 1} entries.
    
    np.random.seed(42)
    
    # Try to find a decomposition with rank < 80
    # Ranks to try: 72, 64, 56, 48
    # For each rank, try multiple attempts.
    # With NumPy check, we can do many more attempts.
    
    target_ranks = [72, 64, 56, 48]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 5:
            break
            
        # Determine number of attempts based on remaining time
        # Estimate time per attempt: ~0.1ms for small matrices?
        # Let's allow up to 5000 attempts if time permits.
        max_attempts = 5000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break
                
            # Generate random sparse matrices
            # Density 0.5 is a good start, or maybe lower for higher rank reduction?
            # Let's stick to {-1, 0, 1} with probability 0.5 for non-zero.
            
            U = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, target_rank))
            
            if check_decomposition_np(U, V, W):
                # Found a valid decomposition
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                result = {"U": U_list, "V": V_list, "W": W_list}
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = result
                # If we found a solution for this rank, we can break to try lower ranks
                break
    
    return best_result
# EVOLVE-BLOCK-END
