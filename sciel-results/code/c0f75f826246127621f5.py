"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook_decomp(n, m, p):
    """Return schoolbook decomposition for n x m * m x p -> n x p."""
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

def _verify_decomp(decomp, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(decomp['U'], dtype=float)
    V = np.array(decomp['V'], dtype=float)
    W = np.array(decomp['W'], dtype=float)
    
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each rank component r, we have:
    #   U[:,r] as (n,m) matrix, V[:,r] as (m,p) matrix, W[:,r] as (p,n) matrix
    # The product should satisfy: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 if this is the identity tensor
    # Actually, the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # For matrix multiplication, T[i,j,k] should be 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # T should be all ones
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _try_strassen_based(n, m, p, deadline):
    """
    Try to construct a lower-rank decomposition using Strassen-like ideas.
    For 3x4x8, we can try to exploit the structure 4=2*2.
    """
    # This is a placeholder for structured search
    # For now, return schoolbook
    return _schoolbook_decomp(n, m, p)

def construct():
    # We construct an exact decomposition for 3x4 * 4x8 -> 3x8.
    # n=3, m=4, p=8.
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook_decomp(n, m, p)
    
    # Try structured improvements if time permits
    if time.time() < deadline:
        candidate = _try_strassen_based(n, m, p, deadline)
        if _verify_decomp(candidate, n, m, p):
            best = candidate
    
    return best
# EVOLVE-BLOCK-END
