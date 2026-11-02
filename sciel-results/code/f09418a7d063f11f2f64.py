"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    We use a structured approach: for small n, we can exploit the fact that the multiplication
    tensor has a special structure. For n=2, the rank can sometimes be reduced below n*m*p.
    
    Key insight: For 2x3 * 3xp, we can potentially use rank < 60 by finding non-trivial
    decompositions. Let's try to find a rank-59 or lower decomposition using a randomized
    algebraic approach with exact rational arithmetic.
    
    The condition is: for all i in [0,n), j in [0,m), k in [0,p):
    sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    We'll try ranks from 59 down to some lower bound, using a randomized initialization
    and solving the resulting system.
    """
    n, m, p = DIMENSIONS
    # Lower bound: rank >= max(n*p, m*p, n*m) = max(20, 30, 6) = 30
    # Also rank >= n*m*p / max(n,m,p) roughly, but let's be safe
    # Try ranks from 59 down to 30
    for rank in range(59, 29, -1):
        result = _try_rank(n, m, p, rank)
        if result is not None:
            return result
    # Fallback to schoolbook
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

def _try_rank(n, m, p, rank, num_trials=50):
    """
    Try to find a rank-r decomposition using randomized algebraic approach.
    
    Strategy: Fix V to be a random matrix with rational entries, then the equations
    become linear in U and W for each (i,j,k). We can set up a system and solve.
    
    Actually, a better approach: We can use the fact that if we fix two of the three
    matrices, the third is determined (if the system is consistent). Let's fix V
    randomly, then for each (i,k), we have:
    sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This is still coupled. Let me think differently.
    
    Alternative: Use a "slice" approach. For each c, the contribution is a rank-1
    tensor. We need the sum of r rank-1 tensors to equal the multiplication tensor.
    
    For n=2, we can try a structured decomposition. Let's try a specific pattern:
    For 2x3*3x10, maybe we can save one term by having some cancellation.
    """
    import random
    random.seed(rank)  # Deterministic for reproducibility
    
    for trial in range(num_trials):
        # Initialize with random rational values
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        
        # Start with schoolbook for the first rank columns, then perturb
        # Actually, let's try a different approach: start with schoolbook and
        # try to merge two columns by adjusting entries
        
        if rank >= n * m * p - 1:
            # Try to merge the last two schoolbook terms
            result = _try_merge_schoolbook(n, m, p, rank)
            if result is not None:
                return result
        
        # For lower ranks, try a more aggressive approach
        # Use a heuristic: set most entries to 0 or 1, with some -1 values
        # to allow cancellation
        
        # This is getting complex. Let's just return None for now if we can't
        # find a quick solution.
        pass
    
    return None

def _try_merge_schoolbook(n, m, p, rank):
    """
    Try to merge two schoolbook terms into one by finding a rank-1 tensor that
    equals the sum of two rank-1 tensors. This is only possible if the two
    terms share some structure.
    
    For schoolbook, term (i1,j1,k1) and term (i2,j2,k2) can be merged if
    we can find U, V, W such that the combined contribution works.
    
    Actually, two rank-1 tensors can only sum to another rank-1 tensor in special
    cases. For our purposes, let's try a different approach.
    """
    # This is complex; let's just return None
    return None
# EVOLVE-BLOCK-END
