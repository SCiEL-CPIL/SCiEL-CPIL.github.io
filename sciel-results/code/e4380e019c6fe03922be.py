"""Rewritten program for 5x5x5 matrix multiplication tensor decomposition."""

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the schoolbook decomposition of 5x5x5 matrix multiplication.
    Rank = 125.
    U: (25, 125)
    V: (25, 125)
    W: (25, 125)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U corresponds to A[i, j] ? 
                # In seed: U[i*m+j]
                # In standard math: A is n x m. Index i (0..n-1), j (0..m-1).
                U[i * m + j][col] = 1
                
                # V corresponds to B[j, k]
                # In seed: V[j*p+k]
                # B is m x p. Index j (0..m-1), k (0..p-1).
                V[j * p + k][col] = 1
                
                # W corresponds to C[i, k] ?
                # In seed: W[k*n+i]
                # C is n x p. Index i (0..n-1), k (0..p-1).
                # Note: The seed uses k*n + i.
                W[k * n + i][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
