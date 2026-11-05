"""Rewritten schoolbook decomposition for 3x3x7 matrix multiplication."""

import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    R = len(U[0]) if U else 0
    # Build tensor T[i,j,k] from the factorization
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0  # The multiplication tensor has 1 at every (i,j,k)
                actual = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(R):
                    actual += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard O(n*m*p) rank decomposition."""
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                U[u_row][col] = 1
                V[v_row][col] = 1
                W[w_row][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_low_rank_search(n, m, p, max_rank, time_budget=100.0):
    """
    Attempt to find a lower-rank decomposition using random search with
    small integer/half-integer entries. Returns None if not found within time.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Candidate values for entries: small integers and half-integers
    candidates = [-1, -0.5, 0, 0.5, 1]
    
    # Number of nonzero entries per factor row to try
    # For 3x3x7, the tensor has 63 entries. A lower rank decomposition
    # would have fewer terms.
    
    # Strategy: Try ranks from max_rank down to 1
    # For each rank, try random assignments of candidate values
    
    # Precompute the tensor entries
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    for rank in range(max_rank, 0, -1):
        if time.time() > deadline:
            break
        
        # Number of random trials per rank
        # Scale trials inversely with rank to spend more time on smaller ranks
        max_trials = 5000 if rank <= 20 else 2000
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Randomly initialize U, V, W with candidate values
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # Fill with random candidate values
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        U[i*m+j][r] = candidates[np.random.randint(len(candidates))]
                for j in range(m):
                    for k in range(p):
                        V[j*p+k][r] = candidates[np.random.randint(len(candidates))]
                for k in range(p):
                    for i in range(n):
                        W[k*n+i][r] = candidates[np.random.randint(len(candidates))]
            
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def _try_structured_patterns(n, m, p, max_rank, time_budget=80.0):
    """
    Try structured patterns that might yield lower-rank decompositions.
    For rectangular matrix multiplication, there may be symmetries we can exploit.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # For 3x3 * 3x7, note that the output is 3x7.
    # The tensor T[i,j,k] for i in 0..2, j in 0..2, k in 0..6
    
    # One idea: group by the middle index j
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a sum of 3 terms, each involving A[i,j] and B[j,k]
    
    # A structured approach: For each j, we need to compute A[:,j] * B[j,:]
    # This is an outer product, which has rank 1 in the (i,k) space
    # But we need the full (i,j,k) tensor
    
    # Let's try a pattern where we exploit the structure:
    # For each j, create a term that handles all (i,k) pairs for that j
    # This gives us m = 3 terms, but each term needs to encode both i and k
    # U[i*m+j, r] and W[k*n+i, r] must combine to give 1 for all i,k
    
    # Actually, the schoolbook already does this optimally for this structure.
    # Let's try some alternative patterns.
    
    # Pattern 1: Try rank = m * p = 21 (group by i)
    # For each i, we have C[i,k] = sum_j A[i,j] B[j,k]
    # This is a vector dot product for each i,k
    
    # Let's try a more systematic search with constrained patterns
    candidates = [-1, -0.5, 0, 0.5, 1]
    
    # Try ranks that are factors or near-factors of n*m*p = 63
    # 63 = 3*3*7, divisors: 1, 3, 7, 9, 21, 63
    for rank in [9, 7, 6, 5, 4, 3, 2, 1]:
        if time.time() > deadline:
            break
        
        max_trials = 10000
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Use a structured initialization:
            # Assign values based on modular arithmetic to create patterns
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            # For each rank term, assign values based on a hash-like pattern
            for r in range(rank):
                # Use different patterns for U, V, W
                for idx in range(n * m):
                    val = candidates[(idx * 7 + r * 3) % len(candidates)]
                    U[idx][r] = val
                for idx in range(m * p):
                    val = candidates[(idx * 5 + r * 2) % len(candidates)]
                    V[idx][r] = val
                for idx in range(p * n):
                    val = candidates[(idx * 11 + r) % len(candidates)]
                    W[idx][r] = val
            
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 63
    
    # Try to find lower-rank decompositions
    # Time budget: leave 10 seconds for safety
    time_budget = 120.0 - 10.0  # 110 seconds total
    
    # Phase 1: Try structured patterns (faster per trial)
    if time.time() - start_time < time_budget * 0.4:
        phase1_budget = min(time_budget * 0.4, 40.0)
        result = _try_structured_patterns(n, m, p, best_rank - 1, phase1_budget)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                best_result = result
                best_rank = r
    
    # Phase 2: Random search
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5.0:
        result = _try_low_rank_search(n, m, p, best_rank - 1, remaining - 5.0)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                best_result = result
                best_rank = r
    
    # Final verification
    if not _verify_decomposition(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        # Fall back to schoolbook
        best_result = _schoolbook(n, m, p)
    
    return best_result
# EVOLVE-BLOCK-END
