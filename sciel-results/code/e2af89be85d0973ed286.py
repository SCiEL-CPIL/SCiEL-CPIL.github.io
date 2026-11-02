"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns U, V, W matrices that decompose the matrix multiplication tensor
    for multiplying a 2x3 matrix by a 3x10 matrix.
    
    The decomposition is exact: for any A (2x3) and B (3x10),
    C[i,k] = sum_r U[i*3 + j, r] * V[j*10 + k, r] * W[k*2 + i, r]
    where the sum is over r and j, with appropriate indexing.
    
    Actually, the standard form is:
    C[i,k] = sum_r U[i, r] * V[r] * W[r, k]  (simplified)
    
    Using the tensor decomposition form:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    For matrix mult (A * B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    T[i,j,k] = delta (the multiplication tensor)
    
    Schoolbook: rank = n*m*p = 2*3*10 = 60
    U[i*m+j, r] = 1 if r corresponds to (i,j,k)
    V[j*p+k, r] = 1 if r corresponds to (i,j,k)  
    W[k*n+i, r] = 1 if r corresponds to (i,j,k)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
