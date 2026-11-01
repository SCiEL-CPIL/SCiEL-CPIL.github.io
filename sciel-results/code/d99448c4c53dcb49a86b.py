"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Build the matrix multiplication tensor T[a, b, c]
    # where a = i*m+j (input A index), b = j*p+k (input B index), c = k*n+i (output index)
    # T[a,b,c] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (k,i) output
    # i.e., T[i*m + j, j*p + k, k*n + i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
    
    # Schoolbook: rank = n*m*p = 42
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.int64)
    V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.int64)
    W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    def verify(U, V, W):
        """Check if U @ diag @ V^T @ ... reconstructs T. 
        Actually: sum_r U[:,r] * V[:,r] * W[:,r] == T
        """
        # Reconstruct: R[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Efficient: for each r, outer product of U[:,r], V[:,r], W[:,r]
        R = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for r in range(U.shape[1]):
            R += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.array_equal(R, T)
    
    # Try to find lower rank decompositions
    # Strategy: search over small integer/half-integer entries
    # For rank r, we need U (6×r), V (21×r), W (14×r)
    # Each entry in {-1, -1/2, 0, 1/2, 1} is a reasonable first attempt
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from lower bound upward
    # Lower bound: at least max(n*m, m*p, p*n) = 21, but tighter bounds exist
    # For matrix mult, rank >= n*m*p / min(n,m,p) roughly, but let's just try
    # Actually a simple bound: rank >= max(n*m, m*p, p*n) is not right
    # The tensor has rank at least the max dimension in any mode
    # But for matrix mult, there are better bounds
    
    # Let's try a structured search: start with schoolbook and try to eliminate terms
    # or search for compact patterns
    
    # Approach: For small ranks, do a randomized/structured search
    # Since entries should be integer or half-integer, work in doubled space (integers)
    # Then divide by 2 at the end
    
    def search_rank(r, time_limit):
        """Try to find a rank-r decomposition with integer/half-integer entries."""
        start = time.time()
        
        # Work with doubled values to keep integers
        # U_d, V_d, W_d have integer entries, and T = sum (U_d[:,r] * V_d[:,r] * W_d[:,r]) / 8
        # So we need sum U_d[:,r] * V_d[:,r] * W_d[:,r] = 8*T
        
        # For rank r, each matrix has r columns
        # U: 6×r, V: 21×r, W: 14×r
        
        # Try structured patterns based on the schoolbook
        # Each schoolbook term corresponds to (i,j,k)
        # Try to group terms
        
        # Method: try all subsets of the 42 schoolbook terms and see if they can be
        # replaced by fewer terms. This is complex, so let's try a different approach.
        
        # Alternative: use the fact that for fixed j, the terms over i,k form a 
        # rank-1 structure in some sense.
        
        # Let's try: for each j, the sub-tensor over (i,k) is outer product of 
        # e_i (in A-space for fixed j) and e_k (in B-space for fixed j) mapped to output.
        # Actually T[i*m+j, j*p+k, k*n+i] = 1 means:
        # For fixed j, we have n*p terms: for each (i,k), one term.
        # The A-index is i*m+j, B-index is j*p+k, W-index is k*n+i.
        
        # For fixed j, consider the n×p sub-problem.
        # A has n entries (i=0..n-1) at position *,j
        # B has p entries (k=0..p-1) at position j,*
        # Output has n*p entries at positions k*n+i
        
        # This is like multiplying an n×1 by 1×p matrix, which has rank n*p in general.
        # But the output pattern k*n+i is specific.
        
        # Let's try a direct combinatorial approach for small r
        # Given the constraints, try r from 21 to 41
        
        for r_try in range(21, min(rank_schoolbook, 35)):
            if time.time() - start > time_limit:
                break
            
            # Try to find a rank r_try decomposition
            # Use a greedy/iterative approach
            # Start with schoolbook and try to merge terms
            
            # For now, try a randomized search with structure
            # Initialize with random integer/half-integer entries
            np.random.seed(42 + r_try)
            
            for attempt in range(50):
                if time.time() - start > time_limit:
                    break
                    
                # Create random candidate with small integer entries
                # U: 6×r_try, V: 21×r_try, W: 14×r_try
                # Entries from {-1, 0, 1} first
                U_cand = np.zeros((n*m, r_try), dtype=np.int64)
                V_cand = np.zeros((m*p, r_try), dtype=np.int64)
                W_cand = np.zeros((p*n, r_try), dtype=np.int64)
                
                # Random sparse initialization
                for c in range(r_try):
                    # Pick random non-zero pattern
                    nz_U = np.random.choice(n*m, size=min(3, n*m), replace=False)
                    nz_V = np.random.choice(m*p, size=min(5, m*p), replace=False)
                    nz_W = np.random.choice(p*n, size=min(4, p*n), replace=False)
                    
                    for idx in nz_U:
                        U_cand[idx, c] = np.random.choice([-1, 1])
                    for idx in nz_V:
                        V_cand[idx, c] = np.random.choice([-1, 1])
                    for idx in nz_W:
                        W_cand[idx, c] = np.random.choice([-1, 1])
                
                if verify(U_cand, V_cand, W_cand):
                    return r_try, U_cand, V_cand, W_cand
        
        return None
    
    # Try structured search for lower ranks
    search_time = 30  # seconds for search
    result = search_rank(21, search_time)
    if result:
        best_rank, best_U, best_V, best_W = result
    
    # If no better found, return schoolbook
    if time.time() > deadline - 5:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
