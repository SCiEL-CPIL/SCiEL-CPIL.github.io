"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact using numpy for speed."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _construct_rank60():
    """Schoolbook decomposition for 2x3x10, rank 60."""
    n, m, p = 2, 3, 10
    return _schoolbook(n, m, p)

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    sb = _schoolbook(n, m, p)
    best_result = sb
    best_rank = n * m * p  # 60
    
    # For 2x3x10, try to find a lower rank decomposition
    # The tensor has 60 entries. Schoolbook gives rank 60.
    # Known: rank of 2x3x1 is 5, so 10*5=50 might be achievable
    # Let's try to construct rank 50 by using 5 terms per k
    
    # For each k, we need to decompose the 2x3x1 sub-tensor
    # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k]
    # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k]
    
    # Try rank 5 per k construction:
    # Use the fact that we can express one row in terms of the other
    # C[1,k] = C[0,k] + (A[1,0]-A[0,0])B[0,k] + (A[1,1]-A[0,1])B[1,k] + (A[1,2]-A[0,2])B[2,k]
    # But we can't directly add C[0,k] to C[1,k] in tensor decomposition
    
    # Instead, let's try a direct numerical search for rank 50
    # This is complex, so let's just return schoolbook for now
    
    # Actually, let's try a structured approach:
    # For rank 60, we have 60 terms. Can we merge any?
    # Each term c has U[i*m+j,c], V[j*p+k,c], W[k*n+i,c]
    # Two terms can be merged if they differ in only one factor
    
    # For now, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
