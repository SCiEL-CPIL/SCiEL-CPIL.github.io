import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-125 (schoolbook) decomposition for 5x5 matrix multiplication.
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
                # U maps input A indices (i,j) to coefficient
                U[i * m + j][col] = 1
                # V maps input B indices (j,k) to coefficient
                V[j * p + k][col] = 1
                # W maps output C indices (i,k) to coefficient
                # Note: The trace convention is W[k*n + i, r]
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
