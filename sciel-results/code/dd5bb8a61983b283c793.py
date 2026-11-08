"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    def to_dict(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Try to find a lower rank decomposition
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # We want to write T as sum_r u_r * v_r * w_r where u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    
    # Let's try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*p, m*p, n*m) = max(24, 32, 12) = 32
    
    lower_bound = max(n * p, m * p, n * m)
    
    # For this small case, let's try to find exact decompositions
    # using a systematic search with rational/integer entries
    
    # Build the target tensor as a 3D array
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # Try ranks from lower_bound to schoolbook
    best_result = None
    best_rank = rank_schoolbook
    
    # Try a few specific ranks
    ranks_to_try = []
    for r in range(lower_bound, rank_schoolbook + 1):
        ranks_to_try.append(r)
        if time.time() > deadline - 30:
            break
    
    # Sort by rank (try smaller first)
    ranks_to_try = sorted(set(ranks_to_try))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try to find an exact decomposition of given rank
        # Use a randomized approach with verification
        
        # For exact decomposition, we can try:
        # 1. Random initialization with rational numbers
        # 2. Check if it exactly reproduces T
        
        # Actually, for exact decompositions, let's use a more structured approach
        # Try the "standard" decomposition patterns
        
        if rank == rank_schoolbook:
            best_result = to_dict(U_schoolbook, V_schoolbook, W_schoolbook)
            best_rank = rank_schoolbook
            continue
        
        # Try a few random attempts for this rank
        max_attempts = 5
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Initialize with small integers
            U = np.random.randint(-2, 3, size=(n * m, rank))
            V = np.random.randint(-2, 3, size=(m * p, rank))
            W = np.random.randint(-2, 3, size=(p * n, rank))
            
            # Check if this decomposition is exact
            # Reconstruct tensor
            T_recon = np.zeros((n * m, m * p, p * n))
            for r in range(rank):
                T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            if np.allclose(T_recon, T, atol=1e-10):
                best_result = to_dict(U, V, W)
                best_rank = rank
                found = True
                break
        
        if found:
            break
    
    # If no better found, use schoolbook
    if best_result is None:
        best_result = to_dict(U_schoolbook, V_schoolbook, W_schoolbook)
    
    return best_result
# EVOLVE-BLOCK-END
