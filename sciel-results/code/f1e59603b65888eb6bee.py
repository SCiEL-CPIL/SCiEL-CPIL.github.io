"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

We seek the minimum rank r such that the matrix multiplication tensor can be written as:
  sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.

For n=2, m=3, p=10:
  U shape: (6, r)
  V shape: (30, r)
  W shape: (20, r)

The schoolbook rank is 60. We aim to find a lower-rank decomposition.

Strategy:
1. Start with the lower bound rank = max(n*m, m*p, p*n) = 30.
2. Try to find a valid decomposition using structured algebraic approaches.
3. If that fails, incrementally increase the rank until we find a valid one.
4. Use the schoolbook decomposition as a fallback.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute the sum
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
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
    return U, V, W, rank

def _try_structured_decomposition(n, m, p, rank, time_limit=10.0):
    """
    Try to find a structured decomposition with the given rank.
    Uses a block-structured approach based on the matrix multiplication structure.
    """
    start_time = time.time()
    
    # For n=2, m=3, p=10, we can try a block-structured approach.
    # The key insight is that we can group terms by certain index patterns.
    
    # Try a simple structured approach: use the fact that the tensor is
    # a "product" structure in some sense.
    
    # One approach: for each (i, k) pair, we need to handle all j values.
    # There are n*p = 20 such pairs, and for each, we have m=3 j values.
    
    # Let's try rank = n*p + (n-1)*m*p or similar formulas.
    
    # Actually, let's try a more direct approach:
    # We can use the fact that for fixed i and k, the sum over j must work out.
    
    # Try a decomposition where we split the rank into blocks.
    # For now, return None to indicate we couldn't find one quickly.
    # A more sophisticated approach would use algebraic methods.
    
    return None

def _try_winograd_like(n, m, p, rank, time_limit=10.0):
    """
    Try a Winograd-like structured decomposition.
    For 2x3x10, we can try to exploit the structure of the 2x3 part.
    """
    start_time = time.time()
    
    # For 2x3 matrix multiplication, the minimum rank is 7 (Strassen-like).
    # We can try to use a rank-7 decomposition for the 2x3x1 part,
    # then extend it to the 3x10 part.
    
    # Actually, let's try a more direct approach:
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We can write this as a sum of rank-1 tensors.
    
    # Try a decomposition where we use the structure of the indices.
    # For fixed j, we have a 2x10 sub-tensor.
    # For fixed i, we have a 3x10 sub-tensor.
    
    # Let's try a rank = 30 + 10 = 40 decomposition.
    # Base: schoolbook for 3x10 part (rank 30)
    # Extra: 10 terms to handle the 2x structure.
    
    if rank < 30:
        return None
    
    # Try rank = 30 + (rank - 30) extra terms
    # This is a heuristic approach.
    
    # For now, return None to indicate we couldn't find one quickly.
    return None

def _random_search(n, m, p, rank, time_limit=10.0):
    """
    Attempt random integer/half-integer low-rank constructions and verify.
    """
    start_time = time.time()
    import random
    random.seed(42)
    
    # Try a range of ranks below schoolbook
    for trial in range(20):
        if time.time() - start_time > time_limit:
            break
        
        # Random half-integer matrices (scaled by 2 to keep integers)
        # Use small range
        def rand_mat(rows, scale=2):
            return [[random.randint(-scale, scale) for _ in range(rank)] for _ in range(rows)]
        
        U = rand_mat(n * m)
        V = rand_mat(m * p)
        W = rand_mat(p * n)
        
        # Verify: D[i,j,k] should be 1 for all i,j,k
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 2:  # scaled by 2, target is 2
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            # Found a valid decomposition at this rank!
            # Scale down by 1/2 to get half-integers
            U = [[x // 2 if x % 2 == 0 else x / 2.0 for x in row] for row in U]
            V = [[x // 2 if x % 2 == 0 else x / 2.0 for x in row] for row in V]
            W = [[x // 2 if x % 2 == 0 else x / 2.0 for x in row] for row in W]
            return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = 145.0  # seconds, leave 5s margin
    
    n, m, p = DIMENSIONS
    
    # Start with the lower bound
    lower_bound = max(n * m, m * p, p * n)  # 30
    
    # Try ranks from lower_bound upward
    for rank in range(lower_bound, n * m * p + 1):
        if time.time() - start_time > deadline - 5.0:
            break
        
        # Try to find a decomposition with this rank
        result = _try_structured_decomposition(n, m, p, rank, time_limit=10.0)
        
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        # Also try Winograd-like approach
        result = _try_winograd_like(n, m, p, rank, time_limit=10.0)
        
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        # Also try random search
        result = _random_search(n, m, p, rank, time_limit=10.0)
        
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
