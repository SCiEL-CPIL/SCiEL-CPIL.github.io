"""Matrix multiplication 3,5,5 - rank-minimal exact tensor decomposition."""

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x5 matrix by a 5x5 matrix.
    
    The tensor T[i,j,k] = 1 for all i in [0,3), j in [0,5), k in [0,5)
    represents the standard matrix multiplication:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We use a block decomposition approach:
    - Split the inner dimension (j) into blocks of size 2 and 1
    - For each block, use the schoolbook decomposition
    - This gives a valid decomposition, though not necessarily rank-minimal
    
    The schoolbook decomposition has rank n*m*p = 3*5*5 = 75.
    We can potentially do better, but 75 is guaranteed correct.
    
    Let me try a smarter approach using the fact that we can combine terms.
    
    Actually, let me implement the schoolbook decomposition correctly.
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Schoolbook decomposition: rank = n * m * p = 75
    rank = n * m * p
    
    # U has shape (n*m, rank) = (15, 75)
    # V has shape (m*p, rank) = (25, 75)  
    # W has shape (p*n, rank) = (15, 75)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j][col] = 1  (coefficient of A[i,j])
                # V[j*p + k][col] = 1  (coefficient of B[j,k])
                # W[k*n + i][col] = 1  (coefficient of C[k*n + i], following trace convention)
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
