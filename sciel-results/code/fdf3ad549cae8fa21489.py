"""Program for matrix multiplication tensor decomposition (3x4x5)."""

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Uses the standard schoolbook algorithm which is rank n*m*p.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We define the decomposition such that for each (i,j,k) in [0,n)x[0,m)x[0,p),
    # there is a rank-1 component with coefficient 1.
    # The index in the decomposition tensor is c = i*m*p + j*p + k (or similar unique mapping).
    # Here we map (i,j,k) -> c using a flat index.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                
                # U row index: i*m + j
                U[i * m + j][c] = 1
                
                # V row index: j*p + k
                V[j * p + k][c] = 1
                
                # W row index: k*n + i
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
