"""Matrix multiplication 2,3,7 - schoolbook decomposition."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-n*m*p (schoolbook) exact tensor decomposition for
    multiplying an n×m matrix by an m×p matrix.
    
    Convention: W[k*n + i, r] is the (i,k) entry of the output factor,
    U[i*m + j, r] and V[j*p + k, r] are the (i,j) and (j,k) entries
    of the input factors.
    
    The multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid triple
    (which it always is for 0<=i<n, 0<=j<m, 0<=k<p), and we decompose it
    as sum_r U[i,j,r] * V[j,k,r] * W[k,i,r].
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
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
# EVOLVE-BLOCK-END
