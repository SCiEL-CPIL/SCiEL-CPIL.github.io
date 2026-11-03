"""
Program for matrix multiplication tensor decomposition.
Task: Find a rank-minimal exact tensor decomposition for 2x5 x 5x6 matrix multiplication.
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions n x m x p.
    
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    Uses the schoolbook decomposition which provides an exact rank-n*m*p decomposition
    with integer entries (1s and 0s).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Populate the schoolbook decomposition
    # The tensor T is sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}
    # Each term corresponds to a rank-1 component.
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i, col j -> flat index i*m + j
                # V index: row j, col k -> flat index j*p + k
                # W index: row k, col i -> flat index k*n + i (trace convention)
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
