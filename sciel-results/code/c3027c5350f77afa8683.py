"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time
import random

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _try_structured_3x3x6(deadline):
    """Try structured decompositions for 3x3x6."""
    n, m, p = 3, 3, 6
    
    # Strategy: Try to find a decomposition where we can share structure
    # between the two 3x3x3 blocks.
    
    # Approach 1: Use a single set of U vectors that work for both halves,
    # but with different V and W vectors.
    
    # For 3x3x3, the minimal known rank is 19 (Alon-Boppana lower bound is 19).
    # But we may not be able to achieve that easily. Let's try rank 27 for 3x3x6
    # by finding a rank-27 decomposition.
    
    # Actually, let's try a different approach: use the fact that we can write
    # the 3x3x6 tensor as a sum of rank-1 tensors.
    
    # Let's try to find a decomposition with rank r where r < 54.
    # We'll use a greedy/structured approach.
    
    # First, let's try to find a 3x3x3 decomposition with small rank,
    # then see if we can combine two such decompositions with shared U.
    
    # For now, let's try a random search with structured initialization.
    
    def _random_search_3x3x6(max_rank, timeout, seed=42):
        """Random search for 3x3x6 decomposition."""
        rng = random.Random(seed)
        start = time.time()
        
        for r in range(max_rank, 53, -1):
            if time.time() > deadline or time.time() - start > timeout:
                break
            
            # Try several random attempts
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                # Initialize with small random values
                U = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n*m)]
                V = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m*p)]
                W = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p*n)]
                
                # Verify
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W, "rank": r}
        
        return None
    
    # Try structured approach: use known 3x3x3 decompositions
    # and combine them with shared U factors.
    
    # For a 3x3x3 decomposition with rank r1, we have U1 (9x r1), V1 (9x r1), W1 (9x r1).
    # For 3x3x6, we need U (9x r), V (18x r), W (18x r).
    
    # If we use the same U for both halves, we need:
    # For k=0,1,2: sum_c U[i,j,c] * V1[j,k,c] * W1[k,i,c] = 1
    # For k=3,4,5: sum_c U[i,j,c] * V2[j,k-3,c] * W2[k-3,i,c] = 1
    
    # This requires U to be the same, which constrains the solution.
    
    # Let's try a different structured approach: use block diagonal structure.
    
    # Actually, let's just try the random search with a reasonable rank range.
    result = _random_search_3x3x6(30, 60, seed=42)
    if result:
        return result
    
    result = _random_search_3x3x6(35, 60, seed=123)
    if result:
        return result
    
    result = _random_search_3x3x6(40, 60, seed=456)
    if result:
        return result
    
    return None

def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    
    # Set a deadline to ensure we finish in time
    deadline = time.time() + 100  # 100 seconds, leaving buffer for the 120s limit
    
    # Try structured search
    result = _try_structured_3x3x6(deadline)
    
    if result and result["rank"] < schoolbook["rank"]:
        # Verify one more time
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Fall back to schoolbook
    return {"U": schoolbook["U"], "V": schoolbook["V"], "W": schoolbook["W"]}
# EVOLVE-BLOCK-END
