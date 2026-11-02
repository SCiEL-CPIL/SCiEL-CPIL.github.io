import numpy as np
import time

DIMENSIONS = [2, 4, 5]
DEADLINE = 145.0

def _build_strassen_36():
    # ... (keep existing) ...
    pass

def _verify_decomposition(U, V, W):
    # ... (keep existing) ...
    pass

def _build_kronecker_matrix(U, V, W, cols=None):
    n, m, p = 2, 4, 5
    if cols is None:
        cols = range(U.shape[1])
    r = len(cols)
    A = np.zeros((n * m * p, r))
    for ci, c in enumerate(cols):
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A[idx, ci] = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    idx += 1
    return A

def _global_rank_reduction(U, V, W, deadline):
    """Try to reduce rank by finding null space of the Kronecker product matrix."""
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    K = _build_kronecker_matrix(U, V, W)
    
    # Check rank
    try:
        s = np.linalg.svd(K, compute_uv=False)
    except:
        return U, V, W
        
    # If rank is effectively lower
    tol = 1e-9
    effective_rank = np.sum(s > tol)
    
    if effective_rank < rank:
        # We can remove (rank - effective_rank) columns
        # Find the null space
        # Use SVD to find null vectors
        U_svd, s, Vt = np.linalg.svd(K, compute_uv=True)
        
        # Null space is the last (rank - effective_rank) rows of Vt
        null_space = Vt[effective_rank:, :]
        
        # We want to remove columns.
        # A simple heuristic: remove the column with the largest norm in the null space?
        # Or just try removing each column and check if rank drops.
        
        # Since we have time, let's try removing columns one by one.
        # We already have _greedy_remove_single, but it's local.
        # This global check confirms if reduction is possible.
        
        # If effective_rank < rank, it means there is a dependency.
        # We can project out the null space.
        # But we need to return a valid decomposition.
        # If we just drop a column, the decomposition might break.
        # We need to re-solve for the remaining columns?
        # No, if we drop a column, the tensor is no longer represented.
        # We need to express the dropped column's contribution as a sum of others.
        # This is exactly what _check_column_redundant does.
        
        # So, if global rank is lower, _greedy_remove_single should have found it.
        # Unless the tolerance was too tight or loose.
        
        # Let's try to be more aggressive with tolerance in _check_column_redundant.
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}
