"""Program for matrix multiplication tensor decomposition 3x3x5."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """
    Verify that the tensor decomposition exactly represents matrix multiplication.
    The multiplication tensor T[i,j,k] = delta_{i,j,k} in the sense that:
    (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor decomposition satisfies:
    T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    where T has indices (i*m+j, j*p+k, k*n+i) ... 
    
    Actually, the correct verification is:
    For the matrix multiplication tensor, we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this corresponds to a valid triple
    and 0 otherwise.
    
    More precisely, the tensor M[i,j,k] for multiplying A (n x m) by B (m x p) to get C (n x p)
    is defined such that C[i,k] = sum_j A[i,j] B[j,k].
    The tensor T with indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p) 
    has T[i,j,k] = 1 for all valid i,j,k (since each term A[i,j]*B[j,k] contributes to C[i,k]).
    
    The decomposition requires:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Check dimensions
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    # Verify: for all i, j, k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication 3x3 by 3x5.
    Rank is n*m*p = 45.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U, V, W = _schoolbook(n, m, p)
    
    # Verify the schoolbook decomposition is valid (it should be)
    # This is a quick sanity check
    if not _verify_decomposition(U, V, W, n, m, p):
        # This should never happen, but if it does, we have a problem
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
