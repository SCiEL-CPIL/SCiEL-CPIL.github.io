"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication.
    T[i,k,j] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i,k,j] = 1 if this is the correct (i,j,k) term for C[i,k] = sum_j A[i,j]B[j,k]
    Actually, the tensor is: for each (i,j,k), the coefficient is 1 in position (i*m+j, j*p+k, k*n+i)
    So we need: for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    And for all other triples (a,b,c) where the product U[a,r]*V[b,r]*W[c,r] contributes,
    the total should be 0 unless (a,b,c) corresponds to a valid (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i.
    
    More precisely: The multiplication tensor T has T[a,b,c] = 1 if there exist i,j,k such that
    a = i*m+j, b = j*p+k, c = k*n+i, and 0 otherwise.
    We need sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Compute the reconstructed tensor
    # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is a 3-tensor. We can compute it as:
    # For each r, the contribution is outer(U[:,r], V[:,r]) * W[:,r] broadcast...
    # Actually: T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # = sum_r (U[:,r] * V[:,r] * W[:,r])_outer... no.
    # It's: for each r, we have a rank-1 tensor U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # So T_recon = sum_r np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Build the target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Check exact match
    if not np.allclose(T_recon, T_target, atol=1e-10):
        return False
    return True

def _schoolbook_decomp(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, n, m, p, max_iters=5000, deadline=None):
    """Try to reduce the rank by removing linearly dependent terms.
    A term r is removable if for all (a,b,c): sum_{r'!=r} U[a,r']*V[b,r']*W[c,r'] = T_target[a,b,c]
    i.e., the contribution of term r is zero everywhere.
    More generally, we can try to find a linear combination of terms that can be eliminated.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Try to remove terms one by one
    for iter_num in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Compute current tensor
        T_current = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        # Check if already valid
        if np.allclose(T_current, T_target, atol=1e-10):
            break
        
        # Try removing each term
        removed = False
        for r in range(rank):
            if deadline and time.time() > deadline:
                break
            
            # Remove term r and check
            U_test = np.delete(U_arr, r, axis=1)
            V_test = np.delete(V_arr, r, axis=1)
            W_test = np.delete(W_arr, r, axis=1)
            
            if U_test.shape[1] == 0:
                T_test = np.zeros((n*m, m*p, p*n), dtype=float)
            else:
                T_test = np.einsum('ar,br,cr->abc', U_test, V_test, W_test)
            
            if np.allclose(T_test, T_target, atol=1e-10):
                # Term r is redundant! Remove it.
                U_arr = U_test
                V_arr = V_test
                W_arr = W_test
                rank -= 1
                removed = True
                break
        
        if not removed:
            # Try to find a linear combination that can be eliminated
            # This is harder. For now, try swapping: replace two terms with one
            # that covers both. This requires solving a system.
            # Skip for now.
            break
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()

def _greedy_construct(n, m, p, deadline=None):
    """Try to construct a low-rank decomposition using a greedy approach.
    Start with schoolbook and try to reduce.
    """
    # Start with schoolbook
    decomp = _schoolbook_decomp(n, m, p)
    U, V, W = decomp["U"], decomp["V"], decomp["W"]
    
    # Try to reduce
    U, V, W = _try_reduce_rank(U, V, W, n, m, p, max_iters=10000, deadline=deadline)
    
    # Verify
    if _verify_decomp(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook_decomp(n, m, p)

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    deadline = t0 + 110.0  # safety margin
    
    # Try the greedy reduction approach
    result = _greedy_construct(n, m, p, deadline=deadline)
    
    # Verify the result
    if not _verify_decomp(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to schoolbook
        result = _schoolbook_decomp(n, m, p)
    
    return result

# EVOLVE-BLOCK-END
