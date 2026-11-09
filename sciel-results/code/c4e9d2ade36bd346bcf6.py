"""Matrix multiplication 3x5x7 rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    # Schoolbook decomposition
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

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school

    def verify(U, V, W, rank):
        """Check if decomposition is exact.
        U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
        We need: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Try structured approaches to find lower rank decompositions
    # Use a greedy peeling approach combined with local search
    
    def try_structured_rank(target_rank, seed=42):
        """Try to find a decomposition of given rank using structured search."""
        np.random.seed(seed)
        
        # Start from schoolbook and try to merge terms
        # Or try to find a sparse pattern that works
        
        # Approach: Use a randomized but structured initialization
        # Based on the observation that for small matrix multiplications,
        # there might be decompositions with some symmetry
        
        # Try different structured patterns
        patterns = [
            "random_sparse",
            "block_diagonal", 
            "cyclic",
            "kron_substitution"
        ]
        
        for pattern in patterns:
            if time.time() > deadline:
                return None
            
            if pattern == "random_sparse":
                # Random sparse initialization with some structure
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                
                # Initialize with random sparsity pattern
                density = 0.3
                for r in range(target_rank):
                    u_mask = np.random.random(n*m) < density
                    v_mask = np.random.random(m*p) < density
                    w_mask = np.random.random(p*n) < density
                    
                    U[u_mask, r] = np.random.choice([-1, 1], size=u_mask.sum())
                    V[v_mask, r] = np.random.choice([-1, 1], size=v_mask.sum())
                    W[w_mask, r] = np.random.choice([-1, 1], size=w_mask.sum())
                
                if verify(U.tolist(), V.tolist(), W.tolist(), target_rank):
                    return (U.tolist(), V.tolist(), W.tolist())
                    
            elif pattern == "kron_substitution":
                # Try Kronecker substitution style
                # This is more structured and might find lower rank
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                
                # For each rank term, assign values based on a pattern
                for r in range(target_rank):
                    # Use different base for each term
                    base = 2 + r
                    for i in range(n):
                        for j in range(m):
                            U[i*m+j, r] = (base ** j) % 2 - 0.5  # This won't work directly
                            # Let's use a simpler pattern
                            U[i*m+j, r] = 1 if (i + j + r) % 2 == 0 else -1
                    
                    for j in range(m):
                        for k in range(p):
                            V[j*p+k, r] = 1 if (j + k + r) % 2 == 0 else -1
                    
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i, r] = 1 if (k + i + r) % 2 == 0 else -1
                
                if verify(U.tolist(), V.tolist(), W.tolist(), target_rank):
                    return (U.tolist(), V.tolist(), W.tolist())
        
        return None

    # Try ranks from just below schoolbook down
    # For 3x5x7, schoolbook is 105
    # Realistic lower bounds suggest we might find something around 90-100
    
    for target_rank in range(rank_schoolbook - 1, max(80, rank_schoolbook - 25), -1):
        if time.time() > deadline:
            break
        
        result = try_structured_rank(target_rank, seed=target_rank)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
