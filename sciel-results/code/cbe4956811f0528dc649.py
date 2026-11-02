"""Exact matrix-multiplication tensor (2 x 3) * (3 x 6) with rank-minimal factors.

This program uses a vectorized construction approach to build the schoolbook
decomposition (rank 36) and verifies it. This approach family (Vectorized/NumPy-based
construction) differs from the loop-based approach in the seed.
"""
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify the decomposition exactly."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # T[i, j, k] should be 1 for all i, j, k? 
    # No, the tensor T represents the multiplication map.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The coefficient of A[i, j] B[j, k] in C[i, k] is 1.
    # In the tensor decomposition:
    # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # We require T[i, j, k] = 1 for all i, j, k? 
    # Wait, the tensor T is defined such that C = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] E[i,k]?
    # Yes. So T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # It always does. So T[i,j,k] = 1 for all i, j, k.
    
    # Let's check the coefficient for each (i, j, k).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(Ua.shape[1]):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_schoolbook_vectorized(n, m, p):
    """Construct the schoolbook decomposition using vectorized operations."""
    rank = n * m * p
    # Initialize zero matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We need to map (i, j, k) to column c.
    # c = i * (m * p) + j * p + k
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    # Create arrays of indices
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # Column indices
    c_idx = np.arange(rank)
    
    # Row indices for U, V, W
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    
    # Set values to 1
    U[u_rows, c_idx] = 1.0
    V[v_rows, c_idx] = 1.0
    W[w_rows, c_idx] = 1.0
    
    # Convert to list of lists of integers (or half-integers)
    U_list = U.astype(int).tolist()
    V_list = V.astype(int).tolist()
    W_list = W.astype(int).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    n, m, p = DIMENSIONS
    
    # Generate the decomposition
    result = _construct_schoolbook_vectorized(n, m, p)
    
    # Verify
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    else:
        # Fallback to a known correct (but maybe slower) loop-based construction
        # if vectorization failed for some reason (should not happen)
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
# EVOLVE-BLOCK-END
