"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook(n, m, p):
    """Return U, V, W as numpy arrays of shape (n*m, rank), (m*p, rank), (p*n, rank)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For schoolbook, T[i,j,k] = 1 for all i,j,k
    # We check: for each c, the outer product structure
    # Actually, the correct check is:
    # For all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    
    # Efficient check: 
    # Let's verify by checking that the decomposition reproduces the identity tensor
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Reshape approach:
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # For a given c, the term is U_c ⊗ V_c ⊗ W_c (outer product)
    # We need sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1 for all i,j,k
    
    # Let's do it directly for small dimensions
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-10):
    """Faster verification using tensor operations."""
    rank = U.shape[1]
    # For each c, compute the contribution
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Let's build it differently. For each c:
    # U_c is a vector of length n*m, V_c is length m*p, W_c is length p*n
    # The tensor from c is: for each (i,j), U_c[i*m+j] * (for each k, V_c[j*p+k] * W_c[k*n+i])
    
    # Actually let's just do it with numpy broadcasting for small sizes
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]  # (n*m,)
        vc = V[:, c]  # (m*p,)
        wc = W[:, c]  # (p*n,)
        
        uc_reshaped = uc.reshape(n, m)  # (n, m)
        vc_reshaped = vc.reshape(m, p)  # (m, p)
        wc_reshaped = wc.reshape(p, n)  # (p, n)
        
        # For each i,j: uc_reshaped[i,j] * sum over k of vc_reshaped[j,k] * wc_reshaped[k,i]
        # = uc_reshaped[i,j] * (vc_reshaped @ wc_reshaped)[j,i]
        prod = vc_reshaped @ wc_reshaped  # (m, n)
        T += uc_reshaped * prod.T  # (n, m) * (n, m).T... wait
        
        # prod is (m, n), prod[j,i] is what we want
        # We want T[i,j] += uc_reshaped[i,j] * prod[j,i]
        T += uc_reshaped * prod.T  # prod.T is (n, m), prod.T[i,j] = prod[j,i]
    
    return np.allclose(T, 1.0, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify schoolbook
    assert _verify_fast(best_U, best_V, best_W, n, m, p), "Schoolbook verification failed"
    
    # Try to reduce rank by removing linearly dependent columns
    # The idea: if some columns are linearly dependent in a way that preserves the tensor,
    # we can remove them.
    
    # Approach: iteratively try to remove columns and check if the decomposition still holds.
    # This is expensive, so we'll use a heuristic.
    
    # For 4x5x6, the known optimal rank is 120 (schoolbook) unless there's a better decomposition.
    # Actually, for matrix multiplication, the rank is known to be at most n*m*p, and for small cases
    # like 4x5x6, it's likely that 120 is optimal or close to it.
    
    # Let's try a simple pruning: check if any column can be removed while preserving the decomposition.
    # This is O(rank^2 * n*m*p) which is manageable for rank=120.
    
    def try_remove_column(U, V, W, idx, n, m, p):
        """Try to remove column idx and see if the decomposition still holds."""
        U_new = np.delete(U, idx, axis=1)
        V_new = np.delete(V, idx, axis=1)
        W_new = np.delete(W, idx, axis=1)
        return _verify_fast(U_new, V_new, W_new, n, m, p)
    
    # Try removing columns one by one
    rank = U.shape[1]
    removed = []
    for idx in range(rank):
        if time.time() > deadline:
            break
        if idx in removed:
            continue
        if try_remove_column(U, V, W, idx, n, m, p):
            U = np.delete(U, idx, axis=1)
            V = np.delete(V, idx, axis=1)
            W = np.delete(W, idx, axis=1)
            removed.append(idx)
            best_rank = U.shape[1]
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Convert to lists for the required output format
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
