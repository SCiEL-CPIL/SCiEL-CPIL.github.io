"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check that U, V, W have the correct shapes
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # For each term r in the decomposition, U[:,r], V[:,r], W[:,r] form a rank-1 tensor
    # The tensor is defined as: T[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent
    # i.e., the (i,j) entry of A multiplies with the (j,k) entry of B to contribute to C[i,k]
    
    # Build the multiplication tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Compute the decomposed tensor
    T_decomp = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T_decomp += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
    
    # Check if they match
    if not np.allclose(T_decomp, T, atol=1e-10):
        return False
    
    return True

def _try_rank(r, n, m, p, time_limit):
    """Try to find a decomposition of rank r using NNLS or similar methods."""
    if time.time() > time_limit:
        return None
    
    N = n * m
    M = m * p
    K = p * n
    
    # The multiplication tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # We want to factor T = sum_r u_r outer(v_r) outer(w_r)
    # where u_r in R^N, v_r in R^M, w_r in R^K
    
    # One approach: fix one set of vectors and solve for others
    # For now, we can try to find a solution using tensor decomposition techniques
    # Since we don't have a direct scipy function for tensor rank decomposition,
    # we can try a greedy or iterative approach
    
    # For this specific problem, let's use the fact that the tensor is sparse
    # and try to find a low-rank approximation
    
    # Actually, for a proper search, we would need to use tensor decomposition
    # algorithms like HOOI, CP decomposition, etc.
    # For now, return None to indicate no improvement found
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with the schoolbook decomposition
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank decompositions
    # For 2x3x4, known lower bounds suggest rank >= 18 or so
    # Try ranks from 18 up to schoolbook_rank - 1
    for r in range(18, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank(r, n, m, p, deadline)
        if result is not None:
            # Verify the result
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r, n, m, p):
                best_rank = r
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                break  # Found the minimal rank in this range
    
    return best_result
# EVOLVE-BLOCK-END
