"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Build the true multiplication tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute decomposed tensor
    D = np.einsum('ar,br,cr->abc', Ua, Va, Wa, optimize=True)
    
    return np.allclose(D, T, atol=1e-9)

def _schoolbook_decomposition():
    """Return the standard schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _try_reduce_rank(schoolbook, target_rank, time_limit=60.0):
    """Try to find a decomposition of target_rank by solving a linear system.
    
    Strategy: Fix W to have a structured form, then solve for U and V.
    """
    n, m, p = DIMENSIONS
    r = target_rank
    start_time = time.time()
    
    # Approach: Try to find U, V, W with integer/half-integer entries
    # by setting up the tensor equations and solving.
    
    # The tensor equation is:
    # For each (a, b, c) in [0,nm) x [0,mp) x [0,pn):
    #   T[a,b,c] = sum_{k=0}^{r-1} U[a,k] * V[b,k] * W[c,k]
    
    # This is trilinear. One approach: fix W, then for each c, 
    # we have a bilinear equation in U and V.
    
    # Alternative: use the fact that if we fix V and W, the equations are linear in U.
    # So we can alternate: fix V,W -> solve for U; fix U,W -> solve for V; fix U,V -> solve for W.
    
    # But this is non-convex and may not converge to an exact solution.
    
    # Let's try a simpler approach: check if we can remove one schoolbook term
    # by seeing if it's in the span of the others.
    
    # Build the matrix whose columns are the rank-1 tensors (flattened)
    nm, mp, pn = n*m, m*p, p*n
    size = nm * mp * pn
    
    # Each schoolbook term is a vector of size 'size'
    # Term (i,j,k) has 1 at position (i*m+j, j*p+k, k*n+i)
    
    # Check linear independence by computing rank of the matrix of these vectors
    # But size = 8*20*10 = 1600, and we have 40 vectors, so this is feasible.
    
    M = np.zeros((size, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i*m+j) * (mp * pn) + (j*p+k) * pn + (k*n+i)
                M[idx, col] = 1.0
                col += 1
    
    # Compute rank of M
    rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    
    if rank_M < n*m*p:
        # There are linear dependencies, so we might be able to reduce rank
        # But this doesn't directly give us a lower-rank decomposition
        pass
    
    # For now, just return None to indicate no improvement found
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook_decomposition()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # Try ranks from best_rank-1 down to max(20, best_rank-10)
    min_possible_rank = max(n*m, m*p, p*n)  # = 20
    
    for target_rank in range(best_rank - 1, min_possible_rank - 1, -1):
        if time.time() > deadline:
            break
        
        result = _try_reduce_rank(best, target_rank, time_limit=30.0)
        if result is not None:
            # Verify the solution
            if _verify_decomposition(result["U"], result["V"], result["W"]):
                best = result
                best_rank = target_rank
                # Don't break; try to go even lower
            else:
                # Invalid solution, continue
                pass
    
    # Final verification
    if not _verify_decomposition(best["U"], best["V"], best["W"]):
        # Fallback to schoolbook
        best = _schoolbook_decomposition()
    
    return best
# EVOLVE-BLOCK-END
