import time
import random
import numpy as np

DIMENSIONS = [6, 6, 6]

def _verify_6x6(U, V, W, n, m, p):
    """
    Vectorized verification of the tensor decomposition.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    Checks if sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1 for all i,j,k.
    """
    # Reshape U, V, W to 3D tensors for easier broadcasting?
    # Or just loop over i, j, k and check the dot product.
    # For R up to 216, looping 216 times with vectorized dot products is fast.
    # U: (36, R) -> (6, 6, R)
    # V: (36, R) -> (6, 6, R)
    # W: (36, R) -> (6, 6, R)
    
    U_3d = U.reshape(n, m, -1)
    V_3d = V.reshape(m, p, -1)
    W_3d = W.reshape(p, n, -1)
    
    # We want T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    # This is a bit tricky to vectorize fully without einsum or loops.
    # einsum: 'ijc,jkc,kic->ijk'
    # Let's try np.einsum.
    
    # U_3d is (n, m, R)
    # V_3d is (m, p, R)
    # W_3d is (p, n, R)
    
    # We need to sum over c.
    # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    
    # Note the indices:
    # U: i, j
    # V: j, k
    # W: k, i
    
    # Let's compute for a fixed i? Or just loop i.
    # Actually, simple loop over i is fast enough.
    result = np.zeros((n, m, p))
    for i in range(n):
        # U_i: (m, R)
        # W_i: (p, R) -> W[:, i, :] is (p, R)
        # We need sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        
        # Let's compute M[j, k] = sum_c (U[i, j, c] * W[k, i, c]) * V[j, k, c]? No.
        # It's a trilinear form.
        
        # Alternative:
        # For each j, k:
        # sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        
        # We can precompute X[j, k, c] = U[i, j, c] * W[k, i, c]
        # Then sum_c X[j, k, c] * V[j, k, c]
        
        # U_i = U[i, :, :] -> (m, R)
        # W_i = W[:, i, :] -> (p, R)
        
        # X[j, k, c] = U_i[j, c] * W_i[k, c]
        # This is an outer product of U_i and W_i, but broadcast over c?
        # U_i[:, np.newaxis, :] * W_i[np.newaxis, :, :] -> (m, p, R)
        
        X = U_i[:, np.newaxis, :] * W_i[np.newaxis, :, :] # (m, p, R)
        
        # Now sum over c: X[j, k, c] * V[j, k, c]
        # V_3d is (m, p, R)
        # So result[i, j, k] = sum_c X[j, k, c] * V_3d[j, k, c]
        
        # This is an element-wise multiply followed by sum over last axis.
        result[i] = np.sum(X * V_3d, axis=-1)
        
    return np.allclose(result, 1.0, atol=1e-9)

def _try_remove_term(U, V, W, idx, n, m, p):
    """
    Try to remove term idx from the decomposition.
    Returns (success, new_U, new_V, new_W)
    """
    R = U.shape[1]
    if R <= 1:
        return False, U, V, W
        
    mask = np.ones(R, dtype=bool)
    mask[idx] = False
    
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    
    if _verify_6x6(U_new, V_new, W_new, n, m, p):
        return True, U_new, V_new, W_new
    
    return False, U, V, W

def _local_search(U, V, W, n, m, p, deadline, rng):
    """
    Greedy local search to minimize rank.
    """
    current_U, current_V, current_W = U, V, W
    current_rank = current_U.shape[1]
    
    # Keep track of indices
    indices = list(range(current_rank))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle indices to try different orders
        rng.shuffle(indices)
        
        # We iterate through the current indices. If we remove one, the list changes.
        # So we should iterate over a copy or handle index shifts.
        # Easier: iterate over range(current_rank) and if removed, adjust.
        
        i = 0
        while i < current_rank and time.time() < deadline:
            # Try to remove term i
            success, nU, nV, nW = _try_remove_term(current_U, current_V, current_W, i, n, m, p)
            
            if success:
                current_U, current_V, current_W = nU, nV, nW
                current_rank -= 1
                # Remove index i from our tracking list? 
                # Actually, we don't strictly need the list if we just iterate 0..R-1.
                # But removing an element shifts the rest.
                # So we don't increment i. We try to remove i again (which is now the next element).
                improved = True
                # To avoid infinite loop on same structure, we can just continue.
                # But if we can't remove i, we increment.
            else:
                i += 1
                
    return current_U, current_V, current_W, current_rank

def _schoolbook_6x6(n, m, p):
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Start with Schoolbook
    U, V, W = _schoolbook_6x6(n, m, p)
    
    # 2. Local Search
    rng = random.Random(42)
    U, V, W, rank = _local_search(U, V, W, n, m, p, deadline, rng)
    
    # 3. Verify final result
    if not _verify_6x6(U, V, W, n, m, p):
        # Fallback
        U, V, W = _schoolbook_6x6(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
