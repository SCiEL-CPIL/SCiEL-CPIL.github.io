"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 145.0  # 5s safety margin
    
    # Build the target multiplication tensor T of shape (n*m, m*p, p*n)
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                valid_indices.append((a, b, c))
    
    def verify(U, V, W):
        """Verify that sum_r U[:,r] V[:,r] W[:,r] == T_target."""
        try:
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            return np.allclose(T_comp, T_target, atol=1e-9)
        except Exception:
            return False
    
    # Trivial schoolbook decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial), dtype=np.float64)
    V_trivial = np.zeros((m * p, rank_trivial), dtype=np.float64)
    W_trivial = np.zeros((p * n, rank_trivial), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, col] = 1.0
                V_trivial[b, col] = 1.0
                W_trivial[c, col] = 1.0
                col += 1
    
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = rank_trivial
    
    # Strategy: Use the tensor's regular structure to find structured low-rank decompositions
    # The tensor T[a,b,c] = 1 iff:
    #   a = i*m + j, b = j*p + k, c = k*n + i for some (i,j,k)
    # This means the non-zero entries form a highly structured pattern
    
    # Approach 1: Block-based decomposition
    # For each i, the sub-tensor involves m x p multiplication
    # We can try to share rank-1 terms across different i values
    
    # Approach 2: Greedy structured search
    # Find rank-1 terms u, v, w such that the outer product u⊗v⊗w 
    # matches a subset of the non-zero entries of T
    
    # Let's try a systematic approach:
    # 1. For each candidate rank R from lower bound to trivial rank
    # 2. Try to find U, V, W with integer/half-integer entries that decompose T
    
    # Lower bound for rank: at least max(n*m, m*p, p*n) = 35
    # But for matrix multiplication, the rank is typically much higher
    
    # Try a structured search with small integer entries
    np.random.seed(42)
    
    # Try candidate ranks
    # For 3x5x7, expect rank between 60 and 105
    candidate_ranks = list(range(60, 105))
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        # Try multiple random initializations with structured patterns
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with small integers in {-1, 0, 1}
            U = np.random.randint(-1, 2, (n * m, target_rank))
            V = np.random.randint(-1, 2, (m * p, target_rank))
            W = np.random.randint(-1, 2, (p * n, target_rank))
            
            if verify(U, V, W):
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                break
        
        if best_rank < rank_trivial:
            break
    
    # If we found a better solution, return it; otherwise return schoolbook
    if best_rank < rank_trivial:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U_trivial.tolist(),
            "V": V_trivial.tolist(),
            "W": W_trivial.tolist()
        }

# EVOLVE-BLOCK-END
