"""Rewritten program for matrix multiplication 3,3,8 rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W decompose the multiplication tensor."""
    # Compute the reconstructed tensor
    # T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_true = _build_tensor(n, m, p)
    return np.allclose(T_recon, T_true, atol=tol)

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_reduce_rank(n, m, p, initial_rank, time_limit=100.0):
    """Try to find a lower rank decomposition by removing columns."""
    start_time = time.time()
    U, V, W = _schoolbook_decomposition(n, m, p)
    current_rank = initial_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Try to remove columns one by one
    # This is a greedy approach - not guaranteed to find optimal
    # but may find some improvements
    
    # Actually, let's try a different approach: use the known 3x3x3 rank 23 result
    # and see if we can extend it
    
    # For now, try a simple heuristic: check if any column is redundant
    # by seeing if it can be expressed as a combination of others
    
    # This is computationally expensive, so let's limit it
    # Try removing each column and checking if the decomposition still holds
    
    for col_to_remove in range(current_rank - 1, -1, -1):
        if time.time() - start_time > time_limit:
            break
        # Try removing this column
        mask = [i for i in range(current_rank) if i != col_to_remove]
        U_test = U[:, mask]
        V_test = V[:, mask]
        W_test = W[:, mask]
        if _verify(U_test, V_test, W_test, n, m, p):
            U, V, W = U_test, V_test, W_test
            current_rank -= 1
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = current_rank
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank
    if time.time() < deadline - 10:
        U2, V2, W2, rank2 = _try_reduce_rank(n, m, p, best_rank, time_limit=90.0)
        if rank2 < best_rank:
            best_U, best_V, best_W = U2, V2, W2
            best_rank = rank2
    
    # Convert to lists
    result = {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
    
    return result
# EVOLVE-BLOCK-END
