"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _validate_decomposition(n, m, p, U, V, W):
    """Validate that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    
    # Check all tensor entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # Should be 1
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check that invalid entries are 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                
                if not (j == j2 and i == i2 and k == k2):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    
    return True

def _schoolbook_decomposition(n, m, p):
    """Returns the schoolbook decomposition with rank n*m*p."""
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
    
    return U, V, W

def _try_structured_search(n, m, p, deadline):
    """
    Search for lower-rank decompositions using structured approaches.
    Returns (U, V, W) if found, else None.
    """
    # Strategy 1: Try rank reduction by grouping
    # For 4x5x6, we can try to find decompositions with rank < 120
    
    # Known bounds: For 4x4x4, best known is 49 (vs 64 schoolbook)
    # For rectangular cases, we need to search
    
    # Approach: Use a structured basis and optimize
    # We'll try various rank values from 119 down
    
    for target_rank in range(119, 80, -1):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition with this rank
        # Using a heuristic: start from schoolbook and try to merge terms
        U, V, W = _schoolbook_decomposition(n, m, p)
        current_rank = n * m * p
        
        # Try random linear combinations to reduce rank
        np.random.seed(42)
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Create random perturbations
            U_mod = np.array(U, dtype=float)
            V_mod = np.array(V, dtype=float)
            W_mod = np.array(W, dtype=float)
            
            # Try to find linear dependencies
            # This is a heuristic approach
            pass
    
    return None

def construct():
    """
    Returns the best matrix multiplication tensor decomposition found.
    Starts with schoolbook and tries to find lower-rank alternatives.
    """
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions
    result = _try_structured_search(n, m, p, deadline)
    if result is not None:
        U, V, W = result
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
