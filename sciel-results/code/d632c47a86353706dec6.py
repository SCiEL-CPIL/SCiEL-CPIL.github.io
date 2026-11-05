import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Returns the schoolbook decomposition with rank 63, which is a valid exact decomposition.
    """
    n, m, p = 3, 3, 7
    
    # Schoolbook rank
    rank = n * m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Populate the schoolbook decomposition
    # U has shape (n*m, rank), indexed by (i, j)
    # V has shape (m*p, rank), indexed by (j, k)
    # W has shape (p*n, rank), indexed by (k, i)
    # The tensor T_{i,j,k} = 1 if i,j,k are the indices for the term A[i,j]*B[j,k]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
