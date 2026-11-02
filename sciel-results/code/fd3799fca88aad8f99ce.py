import time
import numpy as np
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s limit to be safe within 150s
    
    # Schoolbook decomposition (Rank 105)
    rank_sb = n * m * p
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
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Fast verification using NumPy
    def verify_fast(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
        U_reshaped = np.array(U).reshape(n, m, rank)
        V_reshaped = np.array(V).reshape(m, p, rank)
        W_reshaped = np.array(W).reshape(p, n, rank)
        
        # Compute T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Note: W is indexed by k, n, i in the tensor definition W[k*n+i, r]
        # So W_reshaped[k, i, r] corresponds to W[k*n+i, r]
        
        # einsum: 'ijr,jkr,kir->ijk'
        # i: 0..n-1, j: 0..m-1, k: 0..p-1
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            # Check if T is all 1s
            if np.allclose(T, np.ones((n, m, p)), atol=1e-9):
                return True
        except Exception:
            return False
        return False

    # Random search for lower ranks
    random.seed(42)
    np.random.seed(42)
    
    # We will try ranks 104 down to 100
    # Using fast verification, we can do many more trials
    for R in range(best_rank - 1, best_rank - 6, -1):
        if time.time() > deadline - 5:
            break
            
        # Determine number of trials based on remaining time
        # Estimate time per trial: einsum on small arrays is very fast, ~0.1ms?
        # Let's say we can do 1000 trials per second.
        # We want to use up to 5 seconds per rank level if possible.
        max_trials = 1000
        trials_done = 0
        
        while trials_done < max_trials and time.time() < deadline - 5:
            # Random initialization with small integers
            # Using {-1, 0, 1} is a good heuristic for tensor decompositions
            U = np.random.choice([-1, 0, 1], size=(n * m, R))
            V = np.random.choice([-1, 0, 1], size=(m * p, R))
            W = np.random.choice([-1, 0, 1], size=(p * n, R))
            
            # Convert to list for the return format if successful, but verify on numpy
            if verify_fast(U, V, W, R):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            trials_done += 1
            # Small sleep to avoid 100% CPU if it's too fast? No, just loop.
            
    return best_solution
# EVOLVE-BLOCK-END
