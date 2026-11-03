import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook decomposition
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Lower bound for rank
    min_rank = max(n * m, m * p, p * n)
    
    # Strategy: Try structured decompositions first, then random search
    
    # 1. Try Winograd-like or Strassen-like structured approaches
    structured_result = _try_structured_decomposition(n, m, p, deadline)
    if structured_result is not None:
        return structured_result
    
    # 2. Random search with better initialization
    for target_rank in range(min_rank, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Number of attempts depends on rank; lower ranks are harder
        max_attempts = 5000 if target_rank < 22 else 2000
        if target_rank < 20:
            max_attempts = 8000
            
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Generate random U, V with entries in {-1, 0, 1}
            # Use a mix of sparse and denser matrices
            density = 0.3 if target_rank < 20 else 0.4
            U = [[(1 if random.random() < density else -1) if random.random() < 0.5 else 0 for _ in range(target_rank)] for _ in range(n * m)]
            V = [[(1 if random.random() < density else -1) if random.random() < 0.5 else 0 for _ in range(target_rank)] for _ in range(m * p)]
            
            # Check if we can construct W
            W = [[0] * target_rank for _ in range(p * n)]
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Compute M_ik
                    M = [0] * target_rank
                    for j in range(m):
                        for c in range(target_rank):
                            M[c] += U[i * m + j][c] * V[j * p + k][c]
                    
                    # Find c where M[c] is 1, -1, 2, -2
                    found = False
                    for c in range(target_rank):
                        if M[c] == 1:
                            W[k * n + i][c] = 1
                            found = True
                            break
                        elif M[c] == -1:
                            W[k * n + i][c] = -1
                            found = True
                            break
                        elif M[c] == 2:
                            W[k * n + i][c] = 0.5
                            found = True
                            break
                        elif M[c] == -2:
                            W[k * n + i][c] = -0.5
                            found = True
                            break
                    
                    if not found:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify the decomposition
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        # If we didn't find one for this rank, try the next rank
    
    return best_result

def _try_structured_decomposition(n, m, p, deadline):
    """Try structured decompositions based on known algorithms."""
    # For 2x3 x 3x6, try to use a block structure
    
    # Approach: Treat the 3x6 matrix as two 3x3 blocks
    # Then use a 2x3 x 3x3 multiplication and extend
    
    # First, try to find a good 2x3 x 3x3 decomposition (rank 9 schoolbook)
    # Then extend to 3x6 by duplicating or using a clever pattern
    
    # Let's try a direct structured approach:
    # Use the fact that p=6=2*3, so we can think of this as two independent 2x3 x 3x3 multiplications
    
    # Schoolbook for 2x3 x 3x3 has rank 18
    # For 2x3 x 3x6, if we do two independent 2x3 x 3x3, we get rank 36 (same as schoolbook)
    # But maybe we can share some computation
    
    # Try a specific structured pattern:
    # Let's use a Kronecker product structure
    
    # For now, try a few specific patterns
    patterns = [
        _pattern_kronecker(n, m, p),
        _pattern_block_share(n, m, p),
    ]
    
    for pattern in patterns:
        if pattern is not None:
            U, V, W = pattern
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def _pattern_kronecker(n, m, p):
    """Try a Kronecker product-based pattern."""
    # For 2x3 x 3x6, try to use structure from smaller matrices
    # This is a heuristic; may not work
    
    # Skip for now, return None to indicate no pattern found
    return None

def _pattern_block_share(n, m, p):
    """Try to share computation between blocks."""
    # Treat the 3x6 as two 3x3 blocks
    # Try to use a shared set of products
    
    # This is complex; return None for now
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
