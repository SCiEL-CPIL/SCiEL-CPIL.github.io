"""Program for rank-minimal exact tensor decomposition of 2x4 by 4x7 matrix multiplication."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for multiplying a 2x4 matrix by a 4x7 matrix.
    
    The tensor T[i,j,k] (where A is 2x4, B is 4x7, C is 2x7) has:
    - U: shape (n*m, rank) = (8, rank)
    - V: shape (m*p, rank) = (28, rank)  
    - W: shape (p*n, rank) = (14, rank)
    
    We use the schoolbook decomposition of rank 56 = 2*4*7.
    For each (i,j,k) triple, we add a rank-1 term.
    
    Indexing:
    - U is indexed by (i*m + j) rows, so row i*m+j corresponds to A[i,j]
    - V is indexed by (j*p + k) rows, so row j*p+k corresponds to B[j,k]
    - W is indexed by (k*n + i) rows, so row k*n+i corresponds to C[i,k]
    
    The trace convention W[k*n+i, r] means W is indexed with output index k*n+i.
    """
    import time
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin within 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 56
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]   # 8 x 56
    V = [[0] * rank for _ in range(m * p)]   # 28 x 56
    W = [[0] * rank for _ in range(p * n)]   # 14 x 56
    
    # Schoolbook decomposition
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify the decomposition is exact
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j A[i,j] * B[j,k] * 1
    # In tensor form: T[i,j,k] = 1 for all valid (i,j,k)
    # Our decomposition: for each (i,j,k), we have one term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # So the reconstructed tensor at (i,j,k) = sum over all terms where all three match
    # Only the term (i,j,k) matches, giving value 1. Correct.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
