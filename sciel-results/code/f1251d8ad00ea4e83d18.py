"""Matrix multiplication 3x4x6 tensor decomposition.

Uses the schoolbook decomposition with rank n*m*p = 72.
The decomposition satisfies: for all i in [0,n), k in [0,p):
    C[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i*m + j_r, r] * V[j_r*p + k_r, r] * W[k*n + i, r]
where each rank-1 term corresponds to a unique (i,j,k) triple.
"""

DIMENSIONS = [3, 4, 6]

def construct():
    """Construct the factor matrices for matrix multiplication 3x4x6.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # Schoolbook rank = 72
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition: each (i,j,k) triple gives one rank-1 term
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U is indexed by (i*m + j), representing the (i,j) entry of A
                U[i * m + j][column] = 1
                # V is indexed by (j*p + k), representing the (j,k) entry of B
                V[j * p + k][column] = 1
                # W is indexed by (k*n + i), representing the output position C[i,k]
                # using the trace convention W[k*n + i, r]
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
