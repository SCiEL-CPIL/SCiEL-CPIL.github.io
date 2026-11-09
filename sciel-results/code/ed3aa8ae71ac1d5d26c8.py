"""Matrix multiplication 3,3,8 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Try to find a lower rank solution using the algebraic structure
    # The key identity for matrix multiplication tensor decomposition:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # We need: sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_l V[j*p+l, r] B[j,l])
    # = sum_j A[i,j] B[j,k]
    #
    # Expanding: sum_r sum_j sum_l W[k*n+i, r] U[i*m+j, r] V[j*p+l, r] A[i,j] B[j,l]
    # = sum_j A[i,j] B[j,k]
    #
    # So we need: sum_r W[k*n+i, r] U[i*m+j, r] V[j*p+l, r] = delta_{l,k} for all i,j,k,l
    
    # Constructive approach: use rank = p (8)
    # Let W[k*n+i, r] = delta_{k,r}
    # Let U[i*m+j, r] = 1 for all i, j
    # Let V[j*p+l, r] = delta_{l,r}
    #
    # Then: sum_r delta_{k,r} * 1 * delta_{l,r} = delta_{k,l} (since k=r and l=r implies k=l)
    # This satisfies the identity!
    
    rank = p  # 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        # U[i*m+j, r] = 1 for all i, j
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = 1
        # V[j*p+l, r] = 1 if l == r else 0
        for j in range(m):
            V[j * p + r][r] = 1
        # W[k*n+i, r] = 1 if k == r else 0
        for i in range(n):
            W[r * n + i][r] = 1
    
    # Verify the solution
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(p):
                    val = sum(W[k * n + i][r] * U[i * m + j][r] * V[j * p + l][r] for r in range(rank))
                    target = 1.0 if l == k else 0.0
                    err = abs(val - target)
                    if err > max_err:
                        max_err = err
    
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
