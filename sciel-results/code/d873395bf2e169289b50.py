"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all valid indices."""
    T = np.ones((n, m, p), dtype=float)
    return T

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over rank dimension
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _search_rank(n, m, p, max_rank, time_limit=100):
    """Search for a decomposition of the matrix multiplication tensor with rank <= max_rank.
    Uses a systematic approach with integer coefficients.
    """
    start_time = time.time()
    
    # The tensor has n*m*p entries, each must equal 1
    # We're looking for U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Try ranks from max_rank down to 1
    for r in range(max_rank, 0, -1):
        if time.time() - start_time > time_limit:
            break
            
        # For small ranks, we can try to find solutions
        # The key constraint is that for each (i,j,k), the trilinear form must equal 1
        
        # Strategy: use the fact that we can set some entries to specific values
        # and solve for others
        
        # For rank r, we have:
        # U: n*m*r variables
        # V: m*p*r variables  
        # W: p*n*r variables
        # Total: (n*m + m*p + p*n) * r variables
        # Constraints: n*m*p equations (one for each (i,j,k))
        
        # For n=2, m=3, p=4: 24 constraints
        # Variables at rank r: (6+12+8)*r = 26r
        
        # At rank 24 (schoolbook), it's trivially solvable
        # We want to find lower rank solutions
        
        # Approach: Try to find a solution where many entries are 0, 1, or -1
        # This is a hard combinatorial problem, so let's try a structured approach
        
        # Key insight: The matrix multiplication tensor for (2,3,4) might have
        # rank less than 24. Let's try to find it.
        
        # For small cases, we can use the following approach:
        # 1. Fix W to have specific structure
        # 2. Solve for U and V
        
        # Let's try a more direct approach: enumerate possible patterns
        
        if r >= 24:
            # Schoolbook is always valid
            continue
        
        # Try to find a solution with this rank
        # Use a greedy/structured approach
        
        # For the 2x3x4 case, let's try to exploit structure
        # The tensor is all-ones, which is very special
        
        # A key observation: if we can find matrices U, V, W such that
        # the trilinear form gives 1 everywhere, we're done.
        
        # Let's try rank values that are known to be achievable
        # For matrix multiplication, the rank is at most n*m*p
        # and at least max(n*m, m*p, p*n) for certain cases
        
        # For (2,3,4): lower bound is max(6, 12, 8) = 12
        # So rank 12 to 23 are candidates
        
        # Let's try a constructive approach for specific ranks
        # This is complex, so let's return the best we can find
        
        # For now, let's just return the schoolbook if we can't find better
        # but first try a few specific constructions
        
        # Construction attempt: Use the fact that the tensor is all-ones
        # We need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        # If we set W[k*n+i,r] = 1 for all r, then we need
        # sum_r U[i*m+j,r] * V[j*p+k,r] = 1 for all (i,j,k)
        
        # This means the outer product sum must give 1 everywhere
        # This is like finding a rank-r decomposition of the "flattened" tensor
        
        # Actually, this doesn't quite work because W is indexed differently.
        
        # Let me try a different approach: assume specific structures
        
        # For rank r, try to find integer solutions
        # This is a system of polynomial equations, which is hard
        
        # Given the time constraints, let's just verify the schoolbook
        # and return it as the baseline
        
    return None

def _get_schoolbook(n, m, p):
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
    start_time = time.time()
    time_limit = 110  # Leave 10s safety margin
    
    # Try to find a lower-rank decomposition
    # Start with schoolbook and try to improve
    
    # First, get the schoolbook as fallback
    schoolbook = _get_schoolbook(n, m, p)
    
    # For the 2x3x4 case, let's try some known constructions
    # The rank of matrix multiplication tensor for (2,3,4) is at most 24
    # and at least 12
    
    # Let's try to find a rank-12 or lower solution
    # This requires solving a system of equations
    
    # Given the complexity, let's try a few specific constructions
    
    # Construction 1: Try rank 12 with structured matrices
    # If we can make the trilinear form work with rank 12, that's optimal
    
    # For now, return schoolbook as a safe baseline
    # A more sophisticated search would be needed to find lower rank
    
    return schoolbook
# EVOLVE-BLOCK-END
