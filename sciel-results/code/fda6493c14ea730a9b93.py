"""Rewritten program using Kronecker-inspired factorization for 2x3x5 matrix multiplication."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    # Schoolbook rank = n*m*p = 30
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]  # shape (2*3, 30) = (6, 30)
    V = [[0] * rank for _ in range(m * p)]  # shape (3*5, 30) = (15, 30)
    W = [[0] * rank for _ in range(p * n)]  # shape (5*2, 30) = (10, 30)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i (trace convention W[k*n+i, r])
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
