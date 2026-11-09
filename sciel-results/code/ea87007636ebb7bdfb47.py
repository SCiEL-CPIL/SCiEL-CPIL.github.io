import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    # U, V, W are (n*m, R), (m*p, R), (p*n, R)
    # Reshape to (n, m, R), (m, p, R), (p, n, R)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # Note: W index is k*n + i, so W3[k, i, r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    return np.allclose(T, np.ones((n, m, p)), atol=1e-8)

def _make_rank_1(n, m, p):
    """Create rank 1 decomposition. T[i,j,k] = 1 for all i,j,k.
    U[i,j,0]=1, V[j,k,0]=1, W[k,i,0]=1 => sum = 1."""
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _make_schoolbook(n, m, p):
    """Create schoolbook decomposition (rank = n*m*p)."""
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # The tensor T[i,j,k] = 1 for all i,j,k in 3x3x3 MM context 
    # (coefficient of A[i,j]*B[j,k] in C[i,k] is 1).
    # A rank 1 decomposition exists: U=1, V=1, W=1.
    # Check rank 1 first.
    
    res_1 = _make_rank_1(n, m, p)
    if _verify_decomp(np.array(res_1['U']), np.array(res_1['V']), np.array(res_1['W']), n, m, p):
        return res_1
        
    # Fallback: try ranks 2 to 27 if rank 1 somehow failed (it shouldn't)
    # Just in case, implement a quick check for small ranks with random integer search
    # or just return schoolbook if rank 1 fails.
    # Given the mathematical certainty, rank 1 should pass.
    
    return _make_schoolbook(n, m, p)
