import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Currently uses the schoolbook algorithm with rank 125.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook rank is n*m*p
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps input A[i,j] to coefficient c
                U[i * m + j][col] = 1
                # V maps input B[j,k] to coefficient c
                V[j * p + k][col] = 1
                # W maps coefficient c to output C[i,k]
                # Note: The task specifies W[k*n+i, r]
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
