"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def is_exact(U, V, W, n, m, p):
        """Check if the given factors exactly decompose the matrix multiplication tensor."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        
        # Efficient check: compute all tensor entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(r):
                        s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    def try_ranked_decomposition(r, num_trials=50):
        """Try to find a rank-r decomposition using structured random search."""
        for trial in range(num_trials):
            # Use structured random initialization with small integers
            U = np.zeros((n * m, r), dtype=float)
            V = np.zeros((m * p, r), dtype=float)
            W = np.zeros((p * n, r), dtype=float)
            
            for t in range(r):
                # Random entries from {-1, 0, 1} with bias toward sparsity
                U[:, t] = np.random.choice([-1, 0, 1], size=n * m, p=[0.2, 0.6, 0.2])
                V[:, t] = np.random.choice([-1, 0, 1], size=m * p, p=[0.2, 0.6, 0.2])
                W[:, t] = np.random.choice([-1, 0, 1], size=p * n, p=[0.2, 0.6, 0.2])
            
            if is_exact(U, V, W, n, m, p):
                # Convert to list, keeping integer values
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        return None
    
    def try_block_decomposition(r, num_trials=30):
        """Try block-structured decompositions."""
        for trial in range(num_trials):
            U = np.zeros((n * m, r), dtype=float)
            V = np.zeros((m * p, r), dtype=float)
            W = np.zeros((p * n, r), dtype=float)
            
            # Divide rank into blocks and assign patterns
            block_size = max(1, r // 4)
            for block in range(0, r, block_size):
                end = min(block + block_size, r)
                for t in range(block, end):
                    # Each block has a specific structure
                    U[:, t] = np.random.choice([-1, 0, 1], size=n * m)
                    V[:, t] = np.random.choice([-1, 0, 1], size=m * p)
                    W[:, t] = np.random.choice([-1, 0, 1], size=p * n)
            
            if is_exact(U, V, W, n, m, p):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        return None
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But tensor rank can be higher than this bound
    
    # Try to find a better decomposition than schoolbook (rank 60)
    # Try ranks from 30 to 59
    best_result = None
    best_rank = n * m * p  # schoolbook rank
    
    # Try a few promising ranks
    for r in range(30, min(60, best_rank)):
        result = try_ranked_decomposition(r, num_trials=20)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    if best_result is not None:
        return best_result
    
    # Return schoolbook as fallback
    return schoolbook()
# EVOLVE-BLOCK-END
