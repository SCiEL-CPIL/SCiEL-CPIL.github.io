"""
Optimized schoolbook decomposition for 2x3 by 3x5 matrix multiplication.
"""

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps input matrix A (n x m) to intermediate
                # Index in A is i*m + j
                U[i * m + j][col] = 1
                
                # V maps input matrix B (m x p) to intermediate
                # Index in B is j*p + k
                V[j * p + k][col] = 1
                
                # W maps intermediate to output matrix C (n x p)
                # Output index is k*n + i (as per task description W[k*n+i, r])
                W[k * n + i][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
