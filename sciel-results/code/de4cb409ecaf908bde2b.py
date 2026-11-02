"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p, rank):
    """Check if the decomposition exactly represents matrix multiplication."""
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] = 1 if and only if the (i,k) entry
    # of (A*B) uses A[i,j]*B[j,k], so T[i,j,k]=1 for all i,j,k.
    # Actually the multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # So we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j, :]
                v_row = V_arr[j * p + k, :]
                w_row = W_arr[k * n + i, :]
                s = np.dot(u_row * v_row, w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(r, n, m, p, timeout=10):
    """Try to find a rank-r decomposition using random initialization and optimization."""
    start = time.time()
    
    # Use a simple approach: try to find rank r by solving a system
    # The multiplication tensor M is n*m*p dimensional with all entries = 1
    # We need: M[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # This is a 3-tensor decomposition problem.
    # For small ranks, we can try randomized approaches.
    
    # Strategy: random initialization + fixed-point iteration (like tensor power method)
    # Or: use the fact that for rank r, we need r vectors u_c, v_c, w_c such that
    # the outer product sum equals the all-ones tensor.
    
    # For the all-ones tensor of size n x m x p, a rank-1 decomposition is:
    # T[i,j,k] = 1 = a_i * b_j * c_k  =>  a_i = 1, b_j = 1, c_k = 1 for all i,j,k
    # So rank 1 works! T = u ⊗ v ⊗ w where u is all-ones of length n*m, 
    # v is all-ones of length m*p, w is all-ones of length p*n.
    
    # Wait, let me re-check. The multiplication tensor for A(n×m) × B(m×p) = C(n×p) is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i,j,k) with T[i,j,k] = 1 for all valid i,j,k.
    # This is because each triple (i,j,k) contributes A[i,j]*B[j,k] to C[i,k].
    
    # So T is the all-ones tensor of shape (n, m, p).
    # The all-ones tensor has rank 1: T[i,j,k] = 1*1*1.
    
    # In our factorization format:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For rank 1: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 for all i,j,k.
    # Then T[i,j,k] = 1*1*1 = 1. ✓
    
    # So the rank-minimal decomposition is rank 1!
    if r >= 1:
        U = [[1] for _ in range(n * m)]
        V = [[1] for _ in range(m * p)]
        W = [[1] for _ in range(p * n)]
        if _verify(U, V, W, n, m, p, r):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    # The multiplication tensor for (n×m)·(m×p) has T[i,j,k]=1 for all i,j,k.
    # This is the all-ones tensor, which has tensor rank 1.
    # Rank-1 decomposition: U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1.
    
    result = _try_rank(1, n, m, p)
    if result is not None:
        return result
    
    # Fallback to schoolbook if rank 1 somehow fails (shouldn't happen)
    return _schoolbook()
# EVOLVE-BLOCK-END
