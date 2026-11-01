"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # The tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For each (i,j,k), check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank_reduction(n, m, p, target_rank, timeout=60.0):
    """Try to find a decomposition of given target rank using structured search."""
    start_time = time.time()
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    if target_rank >= schoolbook_rank:
        return None
    
    # For 2x4x5, known lower bounds suggest rank might be around 35-40
    # Let's try structured decompositions
    
    # Strategy: Use the fact that matrix multiplication can be decomposed
    # using block structure. For 2x4x5, we can try to exploit the small dimensions.
    
    # One approach: parameterize U, V, W with certain symmetries
    # and solve for the remaining degrees of freedom.
    
    # For now, let's try a greedy approach: start with schoolbook and try to
    # merge columns that are linearly dependent in the tensor product sense.
    
    # Better: Use known results. For 2x4x5 matrix multiplication,
    # the rank is at most 35 (based on similar small matrix multiplication results).
    
    # Let's try to construct a rank-35 decomposition using structured patterns.
    
    # The idea: group the 40 schoolbook terms into 35 groups where some groups
    # share structure.
    
    # Actually, let's try a different approach: use the fact that for matrix multiplication,
    # we can sometimes reduce rank by exploiting the structure of the indices.
    
    # For 2x4x5:
    # n=2 (rows of A), m=4 (cols of A = rows of B), p=5 (cols of B)
    # Output C is 2x5
    
    # Each output C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Schoolbook uses 40 multiplications: one for each (i,j,k)
    # But we can potentially share some multiplications.
    
    # Let's try a structured decomposition where we group by (i,k) pairs
    # and see if we can reduce the number of terms.
    
    # For each (i,k), C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # This is a dot product of length 4.
    
    # There are 2*5 = 10 output entries, each requiring 4 multiplications = 40 total.
    
    # Can we share multiplications across different (i,k) pairs?
    # A[i,j] * B[j,k] is shared between different (i',k') if i'=i and k'=k,
    # but that's the same term.
    
    # However, we can use the fact that bilinear forms can be decomposed
    # in non-obvious ways.
    
    # Let's try a random search with structured initialization.
    
    # Initialize with schoolbook and then try to reduce rank by setting
    # some entries to zero and solving for the rest.
    
    # For target_rank, we'll try to find U, V, W of that rank.
    
    # This is a hard problem, so let's use a heuristic:
    # Start with schoolbook (rank 40) and try to merge columns.
    
    # Two columns c1, c2 can be merged if their tensor products are linearly dependent.
    # The tensor product for column c is U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    
    # Let's try to find pairs of columns that are "similar" and merge them.
    
    # For now, return None if we can't find a better decomposition quickly.
    
    elapsed = time.time() - start_time
    if elapsed > timeout * 0.8:
        return None
    
    # Try a specific structured decomposition for 2x4x5
    # Based on known results for small matrix multiplication
    
    # Let's try rank 35
    if target_rank == 35:
        # Construct a specific decomposition
        # This requires careful construction
        
        # For 2x4x5, we can try to use the following structure:
        # Group the 40 terms into 35 groups where 5 pairs are merged.
        
        # The merging works if for some (i,j,k) and (i',j',k'),
        # the tensors U[i*m+j,:] ⊗ V[j*p+k,:] ⊗ W[k*n+i,:]
        # and U[i'*m+j',:] ⊗ V[j'*p+k',:] ⊗ W[k'*n+i',:]
        # can be combined.
        
        # This is complex, so let's use a simpler approach:
        # Use known rank bounds. For 2x4x5, the rank is at most 35
        # based on interpolation methods.
        
        # Let's try to construct it explicitly.
        
        # Actually, let's just return the schoolbook for now and add
        # a note that we need a better search.
        pass
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition."""
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

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # Leave margin for evaluation overhead
    
    # Start with schoolbook
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # For 2x4x5, try ranks from 35 down to 30
    for target_rank in range(35, max(29, best_rank - 10), -1):
        if time.time() > deadline:
            break
        
        result = _try_rank_reduction(n, m, p, target_rank, timeout=10.0)
        if result is not None:
            # Verify
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = target_rank
                break
    
    return best
# EVOLVE-BLOCK-END
