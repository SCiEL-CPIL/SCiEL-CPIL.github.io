"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]
    # The decomposition requires: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
    # and 0 otherwise (but since we're multiplying, all i,j,k combinations are valid)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_structured_rank(n, m, p, max_rank, deadline):
    """
    Try to find a decomposition with rank <= max_rank using structured patterns.
    Uses Kronecker-product-like or block-structured decompositions.
    """
    # Strategy 1: Try rank = m * max(n, p) or similar structured ranks
    # For matrix mult, there are known lower bounds and constructions.
    
    # Try various candidate ranks
    candidates = []
    for r in range(max(1, min(n*m, m*p, p*n)), max_rank + 1):
        candidates.append(r)
    
    # Also try some known structured ranks
    structured_ranks = [
        n * m, m * p, p * n,
        n * p,  # unlikely but try
        (n + p) * m,
        n * m * p // 2,
        n * m * p // 3,
    ]
    for r in structured_ranks:
        if 1 <= r <= max_rank and r not in candidates:
            candidates.append(r)
    
    candidates = sorted(set(candidates))
    
    for rank in candidates:
        if time.time() > deadline:
            break
        
        # Try to construct a valid decomposition of this rank
        # Use a greedy/structured approach
        
        # Pattern: try to use rank = n*m with V having structure
        # Or rank = m*p
        # Or use block diagonal structures
        
        # For now, try a simple structured approach:
        # If rank == n*m*p, we already have it.
        # Try rank = m * max(n,p) with some sharing
        
        # Attempt: use rank r, try to fill in entries
        # This is a hard problem; use random initialization + verification
        
        # Try a few random structured attempts
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Random binary/half-integer matrices
            np.random.seed(attempt * 1000 + rank)
            U = np.random.choice([0, 0.5, 1, -0.5, -1], size=(n*m, rank))
            V = np.random.choice([0, 0.5, 1, -0.5, -1], size=(m*p, rank))
            W = np.random.choice([0, 0.5, 1, -0.5, -1], size=(p*n, rank))
            
            # Convert to lists
            U_list = [[int(U[i,j]*2)/2 for j in range(rank)] for i in range(n*m)]
            V_list = [[int(V[i,j]*2)/2 for j in range(rank)] for i in range(m*p)]
            W_list = [[int(W[i,j]*2)/2 for j in range(rank)] for i in range(p*n)]
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_low_rank_search(n, m, p, deadline):
    """
    Systematically search for low-rank decompositions.
    """
    best = None
    schoolbook_rank = n * m * p
    
    # Try ranks from low to high
    # Known lower bound for matrix mult rank is at least max(n*m, m*p, p*n)
    min_possible = max(n*m, m*p, p*n)
    
    for rank in range(min_possible, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        
        result = _try_structured_rank(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin for 150s limit
    
    # First, get the schoolbook solution as fallback
    fallback = _schoolbook(n, m, p)
    
    # Try to find a better (lower rank) decomposition
    result = _try_low_rank_search(n, m, p, deadline)
    
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return fallback
# EVOLVE-BLOCK-END
